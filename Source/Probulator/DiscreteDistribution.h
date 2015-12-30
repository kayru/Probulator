#pragma once

#include <random>
#include <vector>

namespace Probulator
{
	// Discrete probability distribution sampling based on alias method
	// http://www.keithschwarz.com/darts-dice-coins
	template <typename T>
	struct DiscreteDistribution
	{
		typedef std::pair<T, size_t> Cell;

		DiscreteDistribution(const T* weights, size_t count, T weightSum)
		{
			std::vector<Cell> large;
			std::vector<Cell> small;
			for (size_t i = 0; i < count; ++i)
			{
				T p = weights[i] * count / weightSum;
				if (p < T(1)) small.push_back({ p, i });
				else large.push_back({ p, i });
			}

			m_cells.resize(count, { T(0), 0 });

			while (large.size() && small.size())
			{
				auto l = small.back(); small.pop_back();
				auto g = large.back(); large.pop_back();
				m_cells[l.second].first = l.first;
				m_cells[l.second].second = g.second;
				g.first = (l.first + g.first) - T(1);
				if (g.first < T(1))
				{
					small.push_back(g);
				}
				else
				{
					large.push_back(g);
				}
			}

			while (large.size())
			{
				auto g = large.back(); large.pop_back();
				m_cells[g.second].first = T(1);
			}

			while (small.size())
			{
				auto l = small.back(); small.pop_back();
				m_cells[l.second].first = T(1);
			}
		}

		size_t operator()(std::mt19937& rng) const
		{
			size_t i = rng() % m_cells.size();
			std::uniform_real_distribution<T> uniformDistribution;
			if (uniformDistribution(rng) <= m_cells[i].first)
			{
				return i;
			}
			else
			{
				return m_cells[i].second;
			}
		}

		std::vector<Cell> m_cells;
	};
}