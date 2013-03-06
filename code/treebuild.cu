#include "Treecode.h"

template<typename real_t, int NLEAF>
void Treecode<real_t, NLEAF>::buildTree()
{
  printf("building tree %d\n", NLEAF);
}

template struct Treecode<float, _NLEAF>;
