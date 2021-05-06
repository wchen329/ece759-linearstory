#ifndef __RANDOMGEN_H__
#define __RANDOMGEN_H__

/* Generic Random Number Generators...
 * using template (partial) specialization for a
 * nice interface
 */

template <class TMP>
class RandomGen
{
	public:
		RandomGen() :
			range(-10, 10),
			rd(),
			gener(rd())
		{
		}

		TMP val()
		{
			return range(gener);
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
			range(-10.0, 10.0),
			rd(),
			gener(rd())
		{
		}

		float val() { return range(gener); }

		std::uniform_real_distribution<float> range;
		std::random_device rd;
		std::mt19937_64 gener;
};


#endif
