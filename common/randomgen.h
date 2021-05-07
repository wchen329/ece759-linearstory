#ifndef __RANDOMGEN_H__
#define __RANDOMGEN_H__
#include <random>

/* Generic Random Number Generators...
 * using template (partial) specialization for a
 * nice interface
 */
template <class TMP>
class RandomGen
{
	public:
		RandomGen() :
			range(-1000, 1000),
			rd(),
			gener(rd())
		{
		}

		TMP operator()() { return val(); }

		TMP val()
		{
			TMP ret = 0;
			while((ret = range(gener)) == 0);
			return ret;
		}

		std::uniform_int_distribution<TMP> range;
		std::random_device rd;
		std::mt19937_64 gener;
};

template <>
class RandomGen<float>
{
	public:
		RandomGen() :
			range(-100.0, 100.0),
			rd(),
			gener(rd())
		{
		}

		float operator()() { return val(); }

		float val()
		{
			float ret = 0;
			while((ret = range(gener)) == 0);
			return ret;
		}

		std::uniform_real_distribution<float> range;
		std::random_device rd;
		std::mt19937_64 gener;
};

#endif
