#pragma once
#include <sys/time.h>

static inline double rtc(void)
{
  struct timeval Tvalue;
  double etime;
  struct timezone dummy;

  //gettimeofday(&Tvalue,NULL);
  gettimeofday(&Tvalue,&dummy);
  etime =  (double) Tvalue.tv_sec +
    1.e-6*((double) Tvalue.tv_usec);
  return etime;
}
