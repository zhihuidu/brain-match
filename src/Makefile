# Compiler and flags
CC = gcc
CFLAGS = -O3 -Wall

# Directories for LAPACK/BLAS (update these if needed)
LAPACK_INCLUDE_DIR ?=  /project/bader/bader/lapack-3.12.0/build/include/
LAPACK_LIB_DIR ?= /project/bader/bader/lapack-3.12.0/build/lib/

# Linker flags for LAPACK, BLAS, and math libraries
LDFLAGS = -L$(LAPACK_LIB_DIR) -llapack -lblas -lgfortran -lm
INCLUDE_FLAGS = -I$(LAPACK_INCLUDE_DIR)

# Targets for each executable
TARGETS = transform-connectome-graph transform-solution convert-VNC-matching \
	score-matching identity-match spectral-match greedy-match refine-matching-hillclimb

# Source files for each program
TRANSFORM_GRAPH_SRCS = transform-connectome-graph.c
TRANSFORM_SOLN_SRCS = transform-solution.c
CONVERT_VNC_SRCS    = convert-VNC-matching.c
SCORE_MATCHING_SRCS = score-matching.c
IDENTITY_MATCH_SRCS = identity-match.c
SPECTRAL_MATCH_SRCS = spectral-match.c
GREEDY_MATCH_SRCS   = greedy-match.c
REFINE_MATCH_HILLCLIMB_SRCS = refine-matching-hillclimb.c


# Default target to build all programs
all: $(TARGETS)

# Transform VNC male and female input graphs
transform-connectome-graph: $(TRANSFORM_GRAPH_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Transform a solution into VNC male and female matching
transform-solution: $(TRANSFORM_SOLN_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Convert the Codex Flywire VNC male and female matching to a generic one
convert-VNC-matching: $(CONVERT_VNC_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Build score-matching
score-matching: $(SCORE_MATCHING_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Build identity-match
identity-match: $(IDENTITY_MATCH_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Build spectral-match with LAPACK and BLAS
spectral-match: $(SPECTRAL_MATCH_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^ $(LDFLAGS)

# Build Greedy match
greedy-match: $(GREEDY_MATCH_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^

# Build refine-matching-hillclimb
refine-matching-hillclimb: $(REFINE_MATCH_HILLCLIMB_SRCS)
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -o $@ $^ -lm

# Clean up build files
clean:
	rm -f $(TARGETS) *.o

# Phony targets to prevent conflicts with files named 'all' or 'clean'
.PHONY: all clean
