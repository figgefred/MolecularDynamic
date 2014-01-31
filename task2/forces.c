#include <omp.h>
#include <stdio.h>
/*
 *  Compute forces and accumulate the virial and the potential
 */
  extern double epot, vir;
  extern double f[];
  extern double x[];
  extern int granularity;

void
forces(int npart, double side, double rcoff)
{

    int t, i, j;
    vir    = 0.0;
    epot   = 0.0;
	int increment = 3;
	int task_increment = increment*granularity;
	int max = npart*increment;
    for (t=0; t < max; t += task_increment) {
		#pragma omp task default(none) private(i,j) firstprivate(t, npart, side, rcoff, increment, task_increment, max)  shared(epot, vir, f, x)
		{
		  // zero force components on particle i 
		  int tmpIndex;
		  double thread_vir = 0.0;
		  double thread_epot = 0.0;
        int t_max = t + task_increment;

			for(i = t;i < t_max; i+=increment)
			{
		     double fxi = 0.0;
		     double fyi = 0.0;
		     double fzi = 0.0;
			// loop over all particles with index > i 
				for (j=i+increment; j<max; j+=increment) 
				{

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


						thread_epot    += rrd3*(rrd3-1.0); 
						thread_vir     -= rd*r148;

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
				tmpIndex = i;
				#pragma omp atomic
	   			f[tmpIndex]     += fxi;
				tmpIndex++;
				#pragma omp atomic
	   			f[tmpIndex]   += fyi;
				tmpIndex++;
				#pragma omp atomic
	   			f[tmpIndex]   += fzi;
			}
		   #pragma omp atomic
			   epot += thread_epot;
		   #pragma omp atomic
			   vir += thread_vir;
		}
	}
	#pragma omp taskwait
}
/*
void
forces(int npart, double side, double rcoff)
{

   int t, c, i, j;
   vir    = 0.0;
   epot   = 0.0;

   if(granularity == 1) {
      forcesRegular(npart, side, rcoff);
      return;
   }

	int increment = 3;
   // Size per task
   int taskSize = granularity;
   // Total amount of tasks with given granularity
   int totalTasks = (npart)/taskSize;
   int max = increment*npart;
   int tHead = 0;
   int tTail = npart*increment - increment;

   for(t = 0; t < totalTasks; t++)
   {
      for(c = 0; c < taskSize; c++)
      {
         if(c % 2 == 0)
         {
            tHead+=increment;
         }            
         else
         {
            tTail-=increment;
         }
      }
      #pragma omp task default(none) firstprivate(npart, side, rcoff, increment, max, taskSize, totalTasks, tHead, tTail) private(t, i, j) shared(epot, vir, x, f, c, granularity)
      {
// zero force components on particle i 
         int tmpIndex;
         double thread_vir = 0.0;
         double thread_epot = 0.0;

			for(t= 0; t < taskSize; t++)
			{
            if(t % 2 == 0)
            {
               i = tHead;
               tHead+=increment;
            }            
            else
            {
               i = tTail;
               tTail-=increment;
            }
//            printf("Thread-%i: TJOOOP woorking with start: %i\n", i);
            double fxi = 0.0;
            double fyi = 0.0;
            double fzi = 0.0;
			// loop over all particles with index > i 
				for (j=i+increment; j<max; j+=increment) 
				{

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


						thread_epot    += rrd3*(rrd3-1.0); 
						thread_vir     -= rd*r148;

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
				tmpIndex = i;
				#pragma omp atomic
	   			f[tmpIndex]     += fxi;
				tmpIndex++;
				#pragma omp atomic
	   			f[tmpIndex]   += fyi;
				tmpIndex++;
				#pragma omp atomic
	   			f[tmpIndex]   += fzi;
			}
		   #pragma omp atomic
			   epot += thread_epot;
		   #pragma omp atomic
			   vir += thread_vir;
      }
   }
}*/
