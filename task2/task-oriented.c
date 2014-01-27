
/*
 *  Compute forces and accumulate the virial and the potential
 */
  extern double epot, vir;

  void
  forces(int npart, double x[], double f[], double side, double rcoff){

    int   i, j;
    vir    = 0.0;
    epot   = 0.0;

    for (i=0; i<npart*3; i+=3) {

#pragma omp task firstprivate(i) private(j) 
{
//#pragma omp for schedule(dynamic) private (j) reduction(+:epot, vir)
//printf("Thread-%i: Reporting in! \n", omp_get_thread_num());
      // zero force components on particle i 

      double fxi = 0.0;
      double fyi = 0.0;
      double fzi = 0.0;
      int tmpIndex;
      double my_vir = 0.0;
      double my_epot = 0.0;

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

          my_epot    += rrd3*(rrd3-1.0);
          my_vir     += rd*r148;

          fxi     += xx*r148;
          fyi     += yy*r148;
          fzi     += zz*r148;

	tmpIndex = j;
	#pragma omp atomic
          f[tmpIndex]    -= xx*r148;
	tmpIndex++;
	#pragma omp atomic
          f[tmpIndex]  -= yy*r148;
	tmpIndex++;
	#pragma omp atomic
          f[tmpIndex]  -= zz*r148;

        }

      }

	
	#pragma omp atomic
          epot    += my_epot;
	#pragma omp atomic
          vir     -= my_vir;

	tmpIndex =i;
      // update forces on particle i 
	#pragma omp atomic
	f[tmpIndex]     += fxi;
	tmpIndex++;
	#pragma omp atomic
	f[tmpIndex]   += fyi;
	tmpIndex++;
	#pragma omp atomic
	f[tmpIndex]   += fzi;

} // END OF TASK

    }

  }
