#include "Treecode.h"

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

  typedef float real_t;
  typedef Treecode<real_t, _NLEAF> Tree;

  Tree tree;

  if (nPtcl > 0)
  {
    fprintf(stdout, "Using Plummer model with nPtcl= %d\n", nPtcl);
    const Plummer data(nPtcl, seed);
    tree.alloc(nPtcl);
    for (int i = 0; i < nPtcl; i++)
    {
      typename Tree::Particle ptclPos, ptclVel;
      ptclPos.x()    = data.pos[i].x;
      ptclPos.y()    = data.pos[i].y;
      ptclPos.z()    = data.pos[i].z;

      ptclVel.x()    = data.vel[i].x;
      ptclVel.y()    = data.vel[i].y;
      ptclVel.z()    = data.vel[i].z;

      ptclVel.mass() = i; //data.mass[i];
      ptclVel.mass() = data.mass[i];
      ptclPos.mass() = data.mass[i];

      tree.h_ptclPos[i] = ptclPos;
      tree.h_ptclVel[i] = ptclVel;
    }
  }
  else
  {
    const ReadTipsy data(fileName);  /* read form stdin */
    nPtcl = data.NTotal;
    tree.alloc(nPtcl);
    for (int i = 0; i < nPtcl; i++)
    {
      typename Tree::Particle pos, vel;
      pos.x() = data. positions[i].x;
      pos.y() = data. positions[i].y;
      pos.z() = data. positions[i].z;
      vel.x() = data.velocities[i].x;
      vel.y() = data.velocities[i].y;
      vel.z() = data.velocities[i].z;
      pos.mass() = vel.mass() = data.positions[i].w;

#if 0
      pos.x() += 0.1234;
      pos.y() += 0.4567;
      pos.z() += 0.5678;
#endif
      tree.h_ptclPos[i] = pos;
      tree.h_ptclVel[i] = vel;
    }
  }
  
#if 1
  {
    double mtot = 0.0;
    typename vec<3,real_t>::type bmin = {+1e10};
    typename vec<3,real_t>::type bmax = {-1e10};
    for (int i = 0; i < nPtcl; i++)
    {
      const Tree::Particle pos = tree.h_ptclPos[i];
      mtot += pos.mass();
      bmin.x = std::min(bmin.x, pos.x());
      bmin.y = std::min(bmin.y, pos.y());
      bmin.z = std::min(bmin.z, pos.z());
      bmax.x = std::max(bmax.x, pos.x());
      bmax.y = std::max(bmax.y, pos.y());
      bmax.z = std::max(bmax.z, pos.z());
    }
    fprintf(stderr, " Total mass = %g \n", mtot);
    fprintf(stderr, "  bmin= %g %g %g \n", bmin.x, bmin.y, bmin.z);
    fprintf(stderr, "  bmax= %g %g %g \n", bmax.x, bmax.y, bmax.z);
  }
#endif

  tree.ptcl_h2d();

  const double t0 = rtc();
  tree.buildTree();
  tree.computeMultipoles();
  tree.makeGroups();
#if 1
  for (int k = 0; k < 1; k++)
  {
    const double2 interactions = tree.computeForces(true);
    fprintf(stderr, " direct= %g   approx= %g \n", interactions.y/tree.nPtcl, interactions.x/tree.nPtcl);
  }
#else
  tree.computeForces();
#endif
  tree.moveParticles();
  tree.computeEnergies();
  const double dt = rtc() - t0;
  fprintf(stderr, " steps are done in %g sec :: rate = %g MPtcl/sec\n",
      dt, tree.nPtcl/1e6/dt);


  return 0;

}
