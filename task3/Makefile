# Makefile for the molecular dynamics code

# for intel compiler on linux
# copy to make.def
CC         = icc
CPP         = icpc
OPTFLAGS   = -openmp -O0


CLINKER     = $(CC)
LIBS        = -lm

CFLAGS    = $(OPTFLAGS)

OBJ=o
EXE=
RM=rm


#
# Object files
#
MDOBJS=    main.$(OBJ) \
	dfill.$(OBJ) \
	domove.$(OBJ) \
      dscal.$(OBJ) \
	fcc.$(OBJ) \
	forces.$(OBJ) \
	mkekin.$(OBJ) \
	mxwell.$(OBJ) \
	prnout.$(OBJ) \
      random.$(OBJ) \
	velavg.$(OBJ) 

#
# Compile
#
md$(EXE):	$(MDOBJS)
	$(CLINKER) $(OPTFLAGS) -o $@ $(MDOBJS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) -c $<

#
# Clean out object files and the executable.
#
clean:
	$(RM) *.$(OBJ) md$(EXE)
