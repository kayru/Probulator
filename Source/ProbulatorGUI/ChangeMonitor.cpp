#include "ChangeMonitor.h"

#include <string>

#ifdef WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

class Win32ChangeMonitor : public ChangeMonitor
{
public:

	Win32ChangeMonitor(const char* path)
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
						modified = true;
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

	HANDLE m_directoryHandle = nullptr;
	bool m_recursive = false;
	OVERLAPPED m_overlapped;
	char m_buffer[65536];
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