/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFF 10000

int main(int argc, char *argv[]) {
    if (argc != 4) {
		printf("Usage: program_name <char_name> <input_file> <output_file>\n");
        return 1;
    }
	
	FILE *in = fopen(argv[2], "rb");
	if (!in) {
        printf("Cant open input file: %s\n", argv[2]);
        return 1;
    }
	
	FILE *out = fopen(argv[3], "w");
	if (!out) {
        printf("Cant open output file: %s\n", argv[3]);
		fclose(in);
        return 1;
    }
	
    fseek(in, 0, SEEK_END); 
    int length = ftell(in);
    fseek(in, 0, SEEK_SET); 

    char* read_buff = (char *)malloc(length); 
    
    if(!fread(read_buff, length, 1, in)) {
        fclose(in);
		fclose(out);
        free(read_buff);
        return 1;
    }

    fclose(in);
	
	
	fwrite("const char* ", 1, 12, out);
	fwrite(argv[1], 1, strlen(argv[1]), out);
	fwrite(" = \n\"", 1, 5, out);
	int i;
	int pos = 0;
	char outbuff[BUFF];
	for(i = 0; i < length; i++) {
		if(read_buff[i] == '\"') {
			outbuff[pos++] = '\\';
			outbuff[pos++] = '\"';
		}
		else if(read_buff[i] == '\n') {
			outbuff[pos++] = '\\';
			outbuff[pos++] = 'n';
			outbuff[pos++] = '"';
			outbuff[pos++] = ' ';
			outbuff[pos++] = '\\';
			outbuff[pos++] = '\n';
			outbuff[pos++] = '"';
		}
		else if(read_buff[i] == '\\') {
			outbuff[pos++] = '\\';
			outbuff[pos++] = '\\';		
		}
		else outbuff[pos++] = read_buff[i];
		
		if(pos > BUFF-10) {
			fwrite(outbuff, 1, pos, out);
			pos = 0;
		}
	}
	fwrite(outbuff, 1, pos, out);
	fwrite("\";\n", 1, 3, out);
	
	
	free(read_buff);
	fclose(out);
	
	return 0;
}
