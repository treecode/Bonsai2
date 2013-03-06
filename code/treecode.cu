#include "anyoption.h"
#include "read_tipsy.h"
#include "cudamem.h"
#include "plummer.h"
#include "Particle4.h"
#include <string>
#include <sstream>

int main(int argc, char * argv[])
{

  std::string fileName = "";
  int seed = 19810614;
  int nPtcl = -1;
  {
    AnyOption opt;
#define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}
		ADDUSAGE(" ");
		ADDUSAGE("Usage");
		ADDUSAGE(" ");
		ADDUSAGE(" -h  --help             Prints this help ");
		ADDUSAGE(" -i  --infile #         Input snapshot filename [tipsy format]");
    ADDUSAGE(" -n  --plummer #        Generate plummer model with a given number of particles");
    ADDUSAGE(" -s  --seed    #        Random seed [" << seed << "]"); 
		ADDUSAGE(" ");
#undef  ADDUSAGE

    opt.setFlag( "help" ,   'h');
    opt.setOption( "infile",  'i');
		opt.setOption( "plummer", 'n' );
		opt.setOption( "seed", 's' );
		
    opt.processCommandArgs( argc, argv );

		if( ! opt.hasOptions() || opt.getFlag("help") || opt.getFlag('h')) 
    {
			opt.printUsage();
			exit(0);
		}
		
		char *optarg = NULL;
    if ((optarg = opt.getValue("plummer"))) nPtcl = atoi(optarg);
    if ((optarg = opt.getValue("seed")))    seed = atoi(optarg);
    if ((optarg = opt.getValue("infile")))  fileName = std::string(optarg);
  }

  if (nPtcl > 0)
  {
    fprintf(stdout, "Using Plummer model with nPtcl= %d\n", nPtcl);
  }
  typedef double real_t;
  typedef Particle4<real_t> Particle;

  host_mem<Particle> h_ptclPos, h_ptclVel;

  double mtot = 0.0;
  if (nPtcl > 0)
  {
    const Plummer data(nPtcl, seed);
    h_ptclPos.alloc(nPtcl);
    h_ptclVel.alloc(nPtcl);
    for (int i = 0; i < nPtcl; i++)
    {
      h_ptclPos[i].x()    = data. pos[i].x;
      h_ptclPos[i].y()    = data. pos[i].y;
      h_ptclPos[i].z()    = data. pos[i].z;
      h_ptclVel[i].x()    = data. vel[i].x;
      h_ptclVel[i].y()    = data. vel[i].y;
      h_ptclVel[i].z()    = data. vel[i].z;
      h_ptclVel[i].mass() = i; //data.mass[i];
      h_ptclVel[i].mass() = data.mass[i];
      h_ptclPos[i].mass() = data.mass[i];
      mtot += data.mass[i];
    }
  }
  else
  {
    const ReadTipsy data(fileName);  /* read form stdin */
    nPtcl = data.NTotal;

    h_ptclPos.alloc(nPtcl);
    h_ptclVel.alloc(nPtcl);

    for (int i = 0; i < nPtcl; i++)
    {
      Particle pos, vel;
      pos.x() = data. positions[i].x;
      pos.y() = data. positions[i].y;
      pos.z() = data. positions[i].z;
      vel.x() = data.velocities[i].x;
      vel.y() = data.velocities[i].y;
      vel.z() = data.velocities[i].z;

      pos.mass() = vel.mass() = data.positions[i].w;
      h_ptclPos[i] = pos;
      h_ptclVel[i] = vel;
      mtot += pos.mass();
    }
  }
  fprintf(stderr, " Total mass = %g \n", mtot);


  return 0;

}
