#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 1024
#define MAX_ID_LENGTH 32

void print_usage(const char *program_name) {
    fprintf(stderr, "Usage: %s <input_matching_file.csv>\n", program_name);
    fprintf(stderr, "Strips 'm' and 'f' prefixes from node IDs and creates generic_matching.csv\n");
}

int main(int argc, char *argv[]) {
    FILE *input, *output;
    char line[MAX_LINE];
    char male_id[MAX_ID_LENGTH], female_id[MAX_ID_LENGTH];
    int line_number = 0;
    
    // Check command line arguments
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Open input file
    input = fopen(argv[1], "r");
    if (!input) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", argv[1]);
        return 1;
    }

    // Create output filename
    char output_filename[MAX_LINE];
    char *dot = strrchr(argv[1], '.');
    if (dot) {
        strncpy(output_filename, argv[1], dot - argv[1]);
        output_filename[dot - argv[1]] = '\0';
        strcat(output_filename, "_generic.csv");
    } else {
        strcpy(output_filename, argv[1]);
        strcat(output_filename, "_generic.csv");
    }

    // Open output file
    output = fopen(output_filename, "w");
    if (!output) {
        fprintf(stderr, "Error: Cannot create output file '%s'\n", output_filename);
        fclose(input);
        return 1;
    }

    // Write generic header
    fprintf(output, "Male ID,Female ID\n");

    // Skip input header
    fgets(line, MAX_LINE, input);
    line_number++;

    // Process each data line
    while (fgets(line, MAX_LINE, input)) {
        line_number++;
        
        // Parse input line, skipping 'm' and 'f' prefixes
        if (sscanf(line, "m%[^,],f%[^\n\r]", male_id, female_id) != 2) {
            fprintf(stderr, "Error on line %d: Malformed input line\n", line_number);
            fprintf(stderr, "Line content: %s", line);
            fclose(input);
            fclose(output);
            return 1;
        }

        // Write to output file without prefixes
        fprintf(output, "%s,%s\n", male_id, female_id);
    }

    printf("Successfully processed %d lines\n", line_number);
    printf("Created %s\n", output_filename);

    fclose(input);
    fclose(output);
    return 0;
}
