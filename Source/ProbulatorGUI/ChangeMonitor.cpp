#include "ChangeMonitor.h"

#include <string>
#include <unordered_map>

#ifdef WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

class Win32ChangeMonitor : public ChangeMonitor
{
public:

	Win32ChangeMonitor(const char* path)
		: m_path(path)
	{
		m_directoryHandle = CreateFile(
			path,
			FILE_LIST_DIRECTORY,
			FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
			0,
			OPEN_EXISTING,
			FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED,
			0);
		memset(&m_overlapped, 0, sizeof(m_overlapped));
		m_overlapped.hEvent = CreateEvent(0, FALSE, FALSE, nullptr);
		startMonitoring();
	}

	~Win32ChangeMonitor()
	{
		CloseHandle(m_directoryHandle);
		CloseHandle(m_overlapped.hEvent);
	}

	std::string getFilename(const PFILE_NOTIFY_INFORMATION& notification)
	{
		int filenameLength = WideCharToMultiByte(CP_UTF8, 0, notification->FileName, notification->FileNameLength / sizeof(wchar_t), 
			nullptr, 0, nullptr, nullptr);

		std::string filename(filenameLength, 0);

		WideCharToMultiByte(CP_UTF8, 0, notification->FileName, notification->FileNameLength / sizeof(wchar_t), 
			&filename[0], filenameLength, nullptr, nullptr);

		return filename;
	}

	u64 getFileTimeStamp(const std::string& path)
	{
		u64 result = 0;

		HANDLE h = CreateFile(path.c_str(), FILE_READ_ATTRIBUTES, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
		if (h == INVALID_HANDLE_VALUE)
		{
			return result;
		}

		FILETIME timeStamp = {};
		GetFileTime(h, nullptr, nullptr, &timeStamp);
		result = (u64(timeStamp.dwHighDateTime) << 32) | timeStamp.dwLowDateTime;

		CloseHandle(h);

		return result;
	}

	virtual bool update() override
	{
		bool modified = false;

		for (;;)
		{
			BOOL success = FALSE;

			DWORD wait_status = WaitForSingleObject(m_overlapped.hEvent, 0);
			if (wait_status == WAIT_OBJECT_0)
			{
				DWORD bytes_transferred = 0;
				success = GetOverlappedResult(m_directoryHandle, &m_overlapped, &bytes_transferred, FALSE);

				PFILE_NOTIFY_INFORMATION notification = (PFILE_NOTIFY_INFORMATION)m_buffer;
				while (bytes_transferred && notification)
				{
					if (notification->Action == FILE_ACTION_MODIFIED)
					{
						std::string filename = getFilename(notification);

						u64 oldTimeStamp = m_timeStamps[filename];
						u64 newTimeStamp = getFileTimeStamp(m_path + "/" + filename);

						modified = modified || newTimeStamp > oldTimeStamp;
						m_timeStamps[filename] = newTimeStamp;
					}
					notification = notification->NextEntryOffset ? (PFILE_NOTIFY_INFORMATION)(m_buffer + notification->NextEntryOffset) : nullptr;
				}

				startMonitoring();
			}

			if (wait_status == WAIT_TIMEOUT)
			{
				break;
			}
		}

		return modified;
	};

private:

	void startMonitoring()
	{
		ReadDirectoryChangesW(
			m_directoryHandle,
			m_buffer,
			sizeof(m_buffer),
			m_recursive,
			FILE_NOTIFY_CHANGE_LAST_WRITE,
			nullptr,
			&m_overlapped,
			nullptr);
	}

	std::string m_path;
	HANDLE m_directoryHandle = nullptr;
	bool m_recursive = false;
	OVERLAPPED m_overlapped;
	char m_buffer[65536];

	std::unordered_map<std::string, u64> m_timeStamps;
};

ChangeMonitor* createChangeMonitor(const char* path)
{
	return new Win32ChangeMonitor(path);
}

#else 

// TODO: MacOS / Linux filesystem change monitor

class DummyChangeMonitor : public ChangeMonitor
{
public:
	virtual bool update() { return false; };
};

ChangeMonitor* createChangeMonitor(const char*)
{
	return nullptr;
}

#endif