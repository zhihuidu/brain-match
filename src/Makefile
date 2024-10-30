# Compiler and flags
CC = gcc
CFLAGS = -O3 -Wall

# Targets for each executable
TARGETS = score-matching identity-match

# Source files for each program
SCORE_MATCHING_SRCS = score-matching.c
IDENTITY_MATCH_SRCS = identity-match.c

# Default target to build all programs
all: $(TARGETS)

# Build score-matching
score-matching: $(SCORE_MATCHING_SRCS)
	$(CC) $(CFLAGS) -o $@ $^

# Build identity-match
identity-match: $(IDENTITY_MATCH_SRCS)
	$(CC) $(CFLAGS) -o $@ $^

# Clean up build files
clean:
	rm -f $(TARGETS) *.o

# Phony targets to prevent conflicts with files named 'all' or 'clean'
.PHONY: all clean
