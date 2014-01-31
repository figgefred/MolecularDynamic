#include <omp.h>
/*
 *  Compute forces and accumulate the virial and the potential
 */
  extern double epot, vir;
  extern double** fs;
  extern int force_length;
  extern int max_threads;

  void
  forces(int npart, double x[], double f[], double side, double rcoff){

    int   i, j;
    vir    = 0.0;
    epot   = 0.0;
	int threadId;

#pragma omp parallel default(none) shared(npart, x, f, side, rcoff, epot, vir, force_length, fs, max_threads) private (i, j, threadId)
{
	threadId = omp_get_thread_num();
#pragma omp for schedule(dynamic) reduction(+:epot, vir)
    for (i=0; i<npart*3; i+=3) {

      // zero force components on particle i 

      double fxi = 0.0;
      double fyi = 0.0;
      double fzi = 0.0;
      double my_epot = 0.0;
      double my_vir = 0.0;
      int tmpIndex;
	  
      // loop over all particles with index > i 
 
      for (j=i+3; j<npart*3; j+=3) {

	// compute distance between particles i and j allowing for wraparound 

        double xx = x[i]-x[j];
        double yy = x[i+1]-x[j+1];
        double zz = x[i+2]-x[j+2];

        if (xx< (-0.5*side) ) xx += side;
        if (xx> (0.5*side) )  xx -= side;
        if (yy< (-0.5*side) ) yy += side;
        if (yy> (0.5*side) )  yy -= side;
        if (zz< (-0.5*side) ) zz += side;
        if (zz> (0.5*side) )  zz -= side;

        double rd = xx*xx+yy*yy+zz*zz;

	// if distance is inside cutoff radius compute forces
	// and contributions to pot. energy and virial 

        if (rd<=rcoff*rcoff) {

          double rrd      = 1.0/rd;
          double rrd3     = rrd*rrd*rrd;
          double rrd4     = rrd3*rrd;
          double r148     = rrd4*(rrd3 - 0.5);

          epot    += rrd3*(rrd3-1.0);
          vir     -= rd*r148;

          fxi     += xx*r148;
          fyi     += yy*r148;
          fzi     += zz*r148;

			tmpIndex = j;
			fs[threadId][tmpIndex]    -= xx*r148;
			tmpIndex++;
			fs[threadId][tmpIndex]  -= yy*r148;
			tmpIndex++;
			fs[threadId][tmpIndex]  -= zz*r148;
        }

      }

	tmpIndex =i;
      // update forces on particle i 
	fs[threadId][tmpIndex]     += fxi;
	tmpIndex++;
	fs[threadId][tmpIndex]   += fyi;
	tmpIndex++;
	fs[threadId][tmpIndex]   += fzi;

    }
	
	#pragma omp for schedule(static) nowait
	// For every 
	for(i = 0; i < force_length; i++)
	{
		for(j = 0; j < max_threads; j++)
		{
			f[i] += fs[j][i];
			fs[j][i] = 0.0;
		}
	}
}
  }
