#include<cstdio>
#include<cstdlib>
#include<cstring>

using namespace std;

char s[1234567];
size_t l;

int main()
{
	FILE *fin = fopen("/input/input.txt","r");
	FILE *fout = fopen("/output/output.txt","w");

	char *line=s;
	while (getline(&line,&l,fin) != -1)
		fprintf(fout,"d%d\n",rand()%2+1);

	fclose(fin);
	fclose(fout);

	return 0;
}
