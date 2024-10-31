#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 1024

void print_usage(const char *program_name) {
    fprintf(stderr, "Usage: %s [-m|-f]\n", program_name);
    fprintf(stderr, "  -m : process male connectome (male_connectome_graph.csv to gm.csv)\n");
    fprintf(stderr, "  -f : process female connectome (female_connectome_graph.csv to gf.csv)\n");
}

int main(int argc, char *argv[]) {
    FILE *input, *output;
    char line[MAX_LINE];
    char from[32], to[32];
    int weight;
    const char *input_filename;
    const char *output_filename;
    char prefix;

    // Check command line arguments
    if (argc != 2 || (strcmp(argv[1], "-m") != 0 && strcmp(argv[1], "-f") != 0)) {
        print_usage(argv[0]);
        return 1;
    }

    // Set filenames based on flag
    if (strcmp(argv[1], "-m") == 0) {
        input_filename = "male_connectome_graph.csv";
        output_filename = "gm.csv";
        prefix = 'm';
    } else {
        input_filename = "female_connectome_graph.csv";
        output_filename = "gf.csv";
        prefix = 'f';
    }

    input = fopen(input_filename, "r");
    if (!input) {
        fprintf(stderr, "Error opening input file: %s\n", input_filename);
        return 1;
    }

    output = fopen(output_filename, "w");
    if (!output) {
        fprintf(stderr, "Error opening output file: %s\n", output_filename);
        fclose(input);
        return 1;
    }

    // Copy header line as is
    if (fgets(line, MAX_LINE, input)) {
        fprintf(output, "%s", line);
    }

    // Create format string with the correct prefix
    char format[32];
    snprintf(format, sizeof(format), "%c%%[^,],%c%%[^,],%%d\n", prefix, prefix);

    // Process data lines
    while (fscanf(input, format, from, to, &weight) == 3) {
        fprintf(output, "%s,%s,%d\n", from, to, weight);
    }

    fclose(input);
    fclose(output);
    
    printf("Successfully processed %s to %s\n", input_filename, output_filename);
    return 0;
}
