/* linsys
 * -
 * This checks include, but also checks doing a small verify of a system.
 */
#include <cassert>
#include "linsys.cuh"

namespace linearstory
{

	template<class DataType> class Verifier : public LinearSystem<DataType>
	{
		public:
			virtual void solve()
			{
			}

		Verifier(size_t n) :
			LinearSystem<DataType>(n)
		{}
	};

}

int main()
{
	// Create a sys with dim 2
	linearstory::Verifier<int> sys(2);

	// Set values of A
	sys[0][0] = 1;
	sys[0][1] = 2;
	sys[1][0] = 2;
	sys[1][1] = 3;

	// Set values of x
	sys.atX(0) = -1;
	sys.atX(1) = 2;

	// Set values of B
	sys.atB(0) = 3;
	sys.atB(1) = 4;

	assert(sys.verify());

	return 0;
}
