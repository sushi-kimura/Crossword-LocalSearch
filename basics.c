/******************** basics.c ********************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "basics.h"

/***** open a file and get the pointer *****/
FILE *openFile( char *fname, char *mode ){
  FILE *fp;
  fp = fopen(fname,mode);
  if(fp==NULL){
    fprintf(stderr,"error: unable to open %s\n",fname);
    exit(EXIT_FAILURE);
  }
  return fp;
}

/***** get the dictionary size *****/
int getDictSize(FILE *fp){
  char w[WORD_MAX];
  int i;
  for(i=0;fgets(w,WORD_MAX,fp)!=NULL;i++)
    1;
  return i;
}

/***** decide whether the dictionary is English or Japanese,
       by first letter *****/
Boolean isEnglish(FILE *fp){
  char x;
  fscanf(fp,"%c",&x);

  printf("<%d>\n",x);

  if(x>=0)
    return True;
  return False;
}

/***** initialize a dictionary *****/
void initDict(Dictionary *D, int n, char *fname){
  FILE *fp;
  int l;
  // set the grid order
  D->n = n;

  // get the dictionary size
  fp = openFile(fname,"r");
  D->size = getDictSize(fp);
  D->m = 0;
  fclose(fp);

  // memory allocation
  D->w = (int**)malloc(D->size*sizeof(int*));
  D->x = (int**)malloc(D->size*sizeof(int*));
  D->len = (int*)malloc(D->size*sizeof(int));
  D->p = (int*)malloc(D->size*sizeof(int));

  // check whether D is English dictionary
  fp = openFile(fname,"r");
  D->en = isEnglish(fp);
  fclose(fp);

  if(D->en == True)
    printf("English\n");
  else
    printf("Japanese\n");

  // initialize variables related to letters
  if(D->en == True)
    D->letters = MAXCODE_EN - MINCODE_EN + 1;
  else
    D->letters = MAXCODE_JA - MINCODE_JA + 1;
  D->dist = (int*)malloc(D->letters*sizeof(int));
  for(l=0;l<D->letters;l++)
    D->dist[l] = 0;
  D->reduce = (int*)malloc(D->letters*sizeof(int));
  D->reds = 0;
  D->inv = (int*)malloc(D->letters*sizeof(int));
}


/***** read the dictionary *****/
void readDict(Dictionary *D, char *fname, int type){
  FILE *fp;
  int l,i,j;
  // read the English/Japanese dictionary
  fp = openFile(fname,"r");
  if (D->en == True) {
    if (type == 0)
      readDictEn(D, fp);
    else
      readDictEnWithWeight(D, fp);
  }
  else
    readDictJa(D,fp);
  fclose(fp);

  // compute extended data
  for(l=0;l<D->letters;l++)
    D->inv[l] = Undef;
  for(l=0;l<D->letters;l++)
    if(D->dist[l]){
      D->reduce[l] = D->reds;
      D->inv[D->reds] = l;
      (D->reds)++;
    }
    else
      D->reduce[l] = Undef;

  for(i=0;i<D->m;i++){
    D->x[i] = (int*)malloc(D->len[i]*sizeof(int));
    for(j=0;j<D->len[i];j++)
      D->x[i][j] = D->reduce[D->w[i][j]];
  }
}


/***** read the English dictionary (in original letters) *****/
void readDictEn(Dictionary *D, FILE *fp){
  char str[WORD_MAX];
  int i=0,l,code;
  while(fscanf(fp,"%s",str) != EOF){
    if(strlen(str)>D->n)
      continue;
    D->w[i] = (int*)malloc(strlen(str)*sizeof(int));
    D->len[i] = strlen(str);
    D->p[i] = 1;
    for(l=0;l<D->len[i];l++){
      code = toupper( str[l] );
      if(code < MINCODE_EN || code > MAXCODE_EN){
	       fprintf( stderr, "error: invalid English dictionary (%s)\n", str );
	       exit(EXIT_FAILURE);
      }
      code = code-MINCODE_EN;
      D->w[i][l] = code;
      D->dist[code]++;
    }
    i++;
    D->m++;
  }
}

void readDictEnWithWeight(Dictionary *D, FILE *fp){
  char str[WORD_MAX], profit[WORD_MAX];
  int i=0,l,code;
  while(fscanf(fp,"%s",str) != EOF){
    /* if(strlen(str)>D->n)
       continue; */
    if(strlen(str)>D->n){
      if(fscanf(fp,"%s",profit)==EOF){ // dummy
	fprintf(stderr, "error: dictionary is strange.\n");
	exit(1);
      }
      continue;
    }
    D->w[i] = (int*)malloc(strlen(str)*sizeof(int));
    D->len[i] = strlen(str);
    /*** read profit ***/
    if(fscanf(fp,"%s",profit)==EOF){
      fprintf(stderr, "error: dictionary is strange.\n");
      exit(1);
    }
    D->p[i] = atof(profit);
    /*******************/
    for(l=0;l<D->len[i];l++){
      code = toupper( str[l] );
      if(code < MINCODE_EN || code > MAXCODE_EN){
	       fprintf( stderr, "error: invalid English dictionary (%s)\n", str );
	       exit(EXIT_FAILURE);
      }
      code = code-MINCODE_EN;
      D->w[i][l] = code;
      D->dist[code]++;
    }
    i++;
    D->m++;
  }
}

/*** 日本語のほうは原口のソースを参考にすること ***/
/***** read the Japanese dictionary (in original letters) *****/
void readDictJa(Dictionary *D, FILE *fp){
  char x,y,z;
  int str[WORD_MAX],i,l,len,code;
  for(i=0;i<D->size;i++){
    len = 0;
    while(fscanf(fp, "%c", &x) != EOF){
      if(x == '\0' || x == '\n')
	break;
      if(getHeadOnes(x) != 3){
	fprintf(stderr, "error: illegal character is found at line %d: x=%c [%d in int]\n",i,x,x);
	exit(EXIT_FAILURE);
      }
      fscanf(fp, "%c", &y);
      fscanf(fp, "%c", &z);
      code = encoder(y,z);
      if(code<MINCODE_JA || code>MAXCODE_JA){
	fprintf(stderr, "error: illegal characeter is found at line %d: code=%d\n",i,code);
	exit(EXIT_FAILURE);
      }
      code = code-MINCODE_JA;
      str[len] = code;
      len++;
    }
    if(len>D->n)
      continue;
    D->w[D->m] = (int*)malloc(len*sizeof(int));
    D->len[D->m] = len;
    D->p[i] = 1;
    for(l=0;l<D->len[D->m];l++){
      D->w[D->m][l] = str[l];
      D->dist[str[l]]++;
    }
    D->m++;
  }
}

/***** get the number of 1's in the head of a given 1-byte char *****/
int getHeadOnes( char c ){
  int ones=0;
  while( c < 0 ){
    c = c<<1;
    ones++;
  }
  return ones;
}

/***** decode an integer to a unicode character (represented by char-string *****/
char *decoder( int p ){
  char *str;
  int t=1,i;
  str = (char*)malloc(4*sizeof(char));
  str[0] = -29;
  for(t=2;t>=1;t--){
    str[t] = 0;
    for(i=0;i<6;i++){
      if( p%2 != 0 )
	str[t] = str[t] | 0x80;
      str[t] = str[t] >> 1;
      str[t] = str[t] & 0x7f;
      p = p >> 1;
    }
    str[t] = str[t] >> 1;
    str[t] = str[t] | 0x80;
  }
  str[3] = '\0';
  return str;
}

/***** encode two 1-byte chars into unicode *****/
int encoder( char x, char y ){
  char z;
  int i,t,v;
  v = 3;
  for(t=0;t<2;t++){
    if( t == 0 )
      z = x;
    else
      z = y;
    z = z << 2;
    for(i=0;i<6;i++){
      v = v << 1;
      if( z < 0 )
	v = v | 1;
      z = z << 1;
    }
  }
  return v;
}

int getConnectedComponents( int n, int **G ){
  int i,j,m=1;
  for(i=0;i<n;i++)
    for(j=0;j<n;j++)
      if( G[i][j] == 0 ){
	       DFS( n, G, i, j, m );
	       m++;
      }

#ifdef DEBUG
  for(i=0;i<n;i++){
    for(j=0;j<n;j++)
      printf(" %2d",G[i][j]);
    printf("\n");
  }
#endif

  return m-1;
}
void DFS( int n, int **G, int i, int j, int m ){
  G[i][j] = m;
  if( i>0 && G[i-1][j] == 0 )
    DFS( n, G, i-1, j, m );
  if( i<n-1 && G[i+1][j] == 0 )
    DFS( n, G, i+1, j, m );
  if( j>0 && G[i][j-1] == 0 )
    DFS( n, G, i, j-1, m );
  if( j<n-1 && G[i][j+1] == 0 )
    DFS( n, G, i, j+1, m );
}
