/******************** LocalSearch.c ********************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

#include "basics.h"

#define True -1
#define False -2

int add(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int sol_size, int t, int **puzzle, int **enable, int *score, int **cover);
int drop(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int sol_size, int **puzzle, int **enable, int *score, int **cover, int t);
int calc(int n, Dictionary *Dict, int **puzzle, int **cover, int mode);
int check_add(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int sol_size, int t, int **puzzle, int **enable);
int check_drop(int n, Dictionary *Dict, Tuple *T, int sol_size, int **cover, int t);
void kick(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int *sol_size, int **puzzle, int **enable, int *score, int **cover);
void shuffle(int arr[], int size);
void display(Dictionary *Dict, int **puzzle, int *score, int sol_size, int n);
int move(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int ****InvT, int **puzzle, int *score, int sol_size, int **enable, int **cover);
int check_move(int n, int **puzzle, int **enable, int **cover, int direction);
int calc_profit(Dictionary *Dict, Tuple *T, int t_size, int *Sol);
void breakConnection(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int *sol_size, int t, int **puzzle, int **enable, int *score, int **cover);

int main(int argc, char *argv[]) {
  Dictionary Dict;
  int div;        // 縦横の添え字
  int n;          // 盤面の次数
  int i;          // 行の添え字
  int j;          // 列の添え字
  int k;          // 単語の添え字
  int d;          // 単語における文字の添え字
  int t = 0;        // タプル集合におけるタプルの添え字
  int code;       // 文字コード
  Tuple *T;       // すべての可能なタプル集合
  int ***InvT[2];   // Tの逆写像 InvT[div][k][i][j] は該当するタプルの T における添字
  int *Sol;       // 解であるタプルの添字集合
  int t_size;     // T に属するタプルの個数
  int sol_size = 0; // 解 Sol に属するタプルの個数
  int profit=0;     //重みの総和
  int moveCount = 0; //moveした回数
  int score = 0;   //目的関数値
  int time1, time2; //処理時間計測用
  int improveCount = 0; //解の改善回数
  float white = 0; //白マス数

  clock_t start = clock();//開始時間

  if (argc < 6) {
    fprintf(stderr, "usage: %s (n)(filename)(seed)(filetype)(movetype)\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  n = atoi(argv[1]);

  int type = atoi(argv[4]); //0：単語のみ　1：重み付き
  if (type != 0 && type != 1) {
    printf("error: filetype only accepts 0 or 1\n");
    exit(EXIT_FAILURE);
  }

  int movetype = atoi(argv[5]); //0:move無し 1:move有り
  if (type != 0 && type != 1) {
    printf("error: movetype only accepts 0 or 1\n");
    exit(EXIT_FAILURE);
  }
  /*** 辞書の初期化 ***/
  initDict(&Dict, n, argv[2]);
  /*** 辞書の読み込み ***/
  readDict(&Dict, argv[2], type);

  /*** InvTの領域を確保 ***/
  for (div=0;div<2;div++) {
    InvT[div] = (int***)malloc(Dict.m*sizeof(int**));
    for (k=0;k<Dict.m;k++) {
      InvT[div][k] = (int**)malloc(Dict.n*sizeof(int*));
      for (i=0;i<Dict.n;i++) {
        InvT[div][k][i] = (int*)malloc(Dict.n*sizeof(int));
      }
    }
  }

  /*** タプルの集合を作る ***/
  T = (Tuple*)malloc(2 * Dict.m*n*n * sizeof(Tuple));
  for (div = 0; div < 2; div++) { // 方向ごとに
    for (k = 0; k < Dict.m; k++) { // 単語ごとに
      int i_max, j_max;
      if (div == 0) {
        i_max = n - Dict.len[k];
        j_max = n - 1;
      }
      else {
        i_max = n - 1;
        j_max = n - Dict.len[k];
      }
      for (i = 0; i <= i_max; i++) { // 行ごとに
        for (j = 0; j <= j_max; j++) { // 列ごとに
          T[t].div = div;
          T[t].k = k;
          T[t].i = i;
          T[t].j = j;
          InvT[div][k][i][j] = t;
          t++;
        }
      }
    }
  }
  t_size = t;

  //パズルの盤面と禁止マスの準備
  int **puzzle;
  int **enable;
  int **tmp_puzzle;
  int **tmp_cover;
  int **tmp_enable;
  int p, q;

  puzzle = (int**)malloc(n * sizeof(int*));
  enable = (int**)malloc(n * sizeof(int*));
  tmp_puzzle = (int**)malloc(n * sizeof(int*));
  tmp_cover = (int**)malloc(n * sizeof(int*));
  tmp_enable = (int**)malloc(n * sizeof(int*));
  for (p = 0; p < n; p++) {
    puzzle[p] = (int*)malloc(n * sizeof(int));
    enable[p] = (int*)malloc(n * sizeof(int));
    tmp_puzzle[p] = (int*)malloc(n * sizeof(int*));
    tmp_cover[p] = (int*)malloc(n * sizeof(int*));
    tmp_enable[p] = (int*)malloc(n * sizeof(int*));
  }

  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      puzzle[p][q] = False;
      enable[p][q] = True;
    }
  }

  //cover配列の準備
  int **cover;
  cover = (int**)calloc(n, sizeof(int*));
  for (p = 0; p < n; p++) {
    cover[p] = (int*)calloc(n, sizeof(int));
  }

  /*** 解の準備 ***/
  Sol = (int*)malloc(t_size * sizeof(int));
  for (t = 0; t < t_size; t++) {
    Sol[t] = False;
  }

  //乱数のシードを設定
  srand(atoi(argv[3]));

  //ランダムにaddするために必要な配列
  int *arr_rand;
  arr_rand = (int*)malloc(t_size * sizeof(int));
  for (p=0;p<t_size;p++) {
    arr_rand[p] = p;
  }
  shuffle(arr_rand,t_size);
  /*** 詰め込めるだけ詰め込む ***/
  int tmp = -1;
  while (tmp != sol_size) {
    tmp = sol_size;
    for (t = 0; t < t_size; t++) {
      sol_size = add(n, &Dict, T, t_size, Sol, sol_size, arr_rand[t], puzzle, enable, &score, cover);
    }
  }
  profit = calc_profit(&Dict, T, t_size, Sol);

  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      if (cover[p][q] != 0) {
        white++;
      }
    }
  }

  clock_t now = clock();//現在時間
  display(&Dict, puzzle, &score, sol_size, n);
  printf("white rate:%f\n", 100 * white / (n*n));
  printf("Processing time : %f[s]\n", (double)(now - start) / CLOCKS_PER_SEC);

  //解を改善する
  int tmp_sol_size, tmp_score, tmp_profit, tmp_white;
  int *tmp_Sol;
  tmp_Sol = (int*)malloc(t_size * sizeof(int));
  int changed = False;

  while (1) {
    //一定時間で終了
    if ((double)(clock()-start)/CLOCKS_PER_SEC > 120) {
      printf("time has come\n");
      break;
    } else if (sol_size==Dict.m) {
      printf("all words are used\n");
      break;
    }

    if (changed == False) {
      //コピー
      tmp_sol_size = sol_size;
      tmp_score = score;
      tmp_profit = profit;
      tmp_white = white;
      for (p = 0; p < t_size; p++) {
        tmp_Sol[p] = Sol[p];
      }
      for (p = 0; p < n; p++) {
        for (q = 0; q < n; q++) {
          tmp_puzzle[p][q] = puzzle[p][q];
          tmp_cover[p][q] = cover[p][q];
          tmp_enable[p][q] = enable[p][q];
        }
      }
    }

    //連結性が崩れるまでdrop
    breakConnection(n, &Dict, T, t_size, Sol, &sol_size, t, puzzle, enable, &score, cover);

    //kickする
    kick(n, &Dict, T, t_size, Sol, &sol_size, puzzle, enable, &score, cover);

    //移動できる方向をランダムで選び、平行移動する
    if (movetype==1) {
      if (move(n, &Dict, T, t_size, Sol, InvT, puzzle, &score, sol_size, enable, cover) == True) {
        moveCount++;
      }
    }

    //詰められるだけadd
    shuffle(arr_rand,t_size);
    tmp = -1;
    while (tmp != sol_size) {
      tmp = sol_size;
      for (t = 0; t < t_size; t++) {
        sol_size = add(n, &Dict, T, t_size, Sol, sol_size, arr_rand[t], puzzle, enable, &score, cover);
      }
    }

    profit = calc_profit(&Dict, T, t_size, Sol);

    white = 0;
    for (p = 0; p < n; p++) {
      for (q = 0; q < n; q++) {
        if (cover[p][q] != 0) {
          white++;
        }
      }
    }

    //改善していれば繰り返す
    if ((white > tmp_white) || (white == tmp_white && score >= tmp_score)) {
      changed = False;
      //改善解の表示
      if (white > tmp_white) {
        improveCount++;
      }
      printf("--- improved[%d] ---\n\n", improveCount);
      display(&Dict, puzzle, &score, sol_size, n);
      now = clock();//現在時間
      printf("white rate:%f\n", 100 * white / (n*n));
      printf("Processing time : %f[s]\n", (double)(now - start) / CLOCKS_PER_SEC);
    }
    //改善していなければパズルをもとに戻す
    else {
      changed = False;
      sol_size = tmp_sol_size;
      score = tmp_score;
      profit = tmp_profit;
      white = tmp_white;
      for (p = 0; p < t_size; p++) {
        Sol[p] = tmp_Sol[p];
      }
      for (p = 0; p < n; p++) {
        for (q = 0; q < n; q++) {
          puzzle[p][q] = tmp_puzzle[p][q];
          cover[p][q] = tmp_cover[p][q];
          enable[p][q] = tmp_enable[p][q];
        }
      }
    }
  }

  //最終的なpuzzleの表示
  printf("--- COMPLETE ---\n\n");
  display(&Dict, puzzle, &score, sol_size, n);
  printf("profit:%d\n", profit);
  now = clock();//現在時間
  printf("Processing time:%f[s]\n", (double)(now - start) / CLOCKS_PER_SEC);
  //moveした回数
  printf("move:%d\n", moveCount);
  //白マスの割合
  printf("white rate:%f%%\n", 100 * white / (n*n));
  //メモリの解放
  free(T);
  for (p = 0; p < n; p++) {
    free(puzzle[p]);
    free(enable[p]);
    free(tmp_puzzle[p]);
    free(tmp_cover[p]);
  }
  free(puzzle);
  free(enable);
  free(tmp_puzzle);
  free(tmp_cover);
  free(Sol);
  free(arr_rand);
  return 0;
}


/******************** add **********************/

int add(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int sol_size, int t, int **puzzle, int **enable, int *score, int **cover) {
  int p, q, k;
  //int *haitteta;
  int canAdd;
  int length = Dict->len[T[t].k];


  //単語が詰め込み可能であるかどうか
  canAdd = check_add(n, Dict, T, t_size, Sol, sol_size, t, puzzle, enable);
  if (canAdd == False) {
    return sol_size;
  }

  //盤面に配置
 // haitteta = (int*)malloc(length*sizeof(int));
  for (p = 0; p < length; p++) {
    if (T[t].div == 0) {
      //haitteta[p] = puzzle[T[t].i+p][T[t].j];
      puzzle[T[t].i + p][T[t].j] = Dict->x[T[t].k][p];
      //printf("<p=%d, %d>",p,Dict->x[T[t].k][p]);
    } else {
      //haitteta[p] = puzzle[T[t].i][T[t].j + p];
      puzzle[T[t].i][T[t].j + p] = Dict->x[T[t].k][p];
    }
  }
  /*
  //解が改善しなかった場合は終了
  int newScore = calc(n, Dict, puzzle, cover, 0);
  if (newScore < *score) {
    for (k = 0; k < length; k++) {
      if (T[t].div == 0) {
        puzzle[T[t].i + k][T[t].j] = haitteta[k];
      } else {
        puzzle[T[t].i][T[t].j + k] = haitteta[k];
      }
    }
    goto END;
  }
  */
  //盤面に配置した場合、禁止マスを設定
  if (T[t].div == 0) {
    if (T[t].i - 1 >= 0) {
      enable[T[t].i - 1][T[t].j] = False;
    }
    if (T[t].i + length < n) {
      enable[T[t].i + length][T[t].j] = False;
    }
  } else {
    if (T[t].j - 1 >= 0) {
      enable[T[t].i][T[t].j - 1] = False;
    }
    if (T[t].j + length < n) {
      enable[T[t].i][T[t].j + length] = False;
    }
  }

  //cover配列を更新
  if (T[t].div == 0) {
    for (p = 0; p < length; p++) {
      cover[T[t].i + p][T[t].j]++;
    }
  } else {
    for (p = 0; p < length; p++) {
      cover[T[t].i][T[t].j + p]++;
    }
  }

  //cover配列の表示
  /*
  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      printf("%d ", cover[p][q]);
    }
    printf("\n");
  }
  */

  //カバー配列の表示
 /*
  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      if (enable[p][q] == False) {
        printf("F ");
      } else {
        printf("T ");
      }
    }
    printf("\n");
  }
*/

  // 詰め込みが可能と判定された場合:
  Sol[t] = True;
  *score = calc(n, Dict, puzzle, cover, 0);

  //puzzleの表示
  //printf("--- added ---\n\n");
  //display(Dict, puzzle, score, sol_size+1, n);

  return sol_size+1;

END:
  //free(haitteta);
  return sol_size;
}


/******************** calc **********************/
//mode=0のときはcover配列上の値の合計を返す。
int calc(int n, Dictionary *Dict, int **puzzle, int **cover, int mode) {
  int p, q, r, s;
  int score=0;

  if (mode == 0) {
    for (p = 0; p < n; p++) {
      for (q = 0; q < n; q++) {
        score += cover[p][q];
      }
    }
    return score;
  }

  int *word;
  int length=0;
  int matching = False;
  int inDict=0;//パズルにある単語で辞書内にある単語と一致する数
  int penalty=0;

  word = (int*)malloc(n * sizeof(int));

  //文字列が辞書内にあるかを調べる
  //縦
  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q ++) {
      if ((p == 0 && puzzle[p][q] != False) || (p > 0 && puzzle[p - 1][q] == False && puzzle[p][q] != False)) {
        for (r=0;r<n-p;r++) {
          if (puzzle[p+r][q]== False) {
            break;
          } else {
            word[r] = puzzle[p+r][q];
            length++;
          }
        }
        /*debug
        for (r=0;r<length;r++) {
          printf("word[%d] = %d ",r,word[r]);
        }
        printf("\n");
        printf("length=%d\n",length);
        */

        //wordが辞書内にあるかをチェック
        for (r=0; r<Dict->m; r++) {
          if (length == Dict->len[r]) {
            for (s=0; s<length; s++) {
              if (word[s] == Dict->x[r][s]) {
                matching = True;
              } else {
                matching = False;
                break;
              }
            }
          }
          if (matching == True) {
            inDict += length;
            break;
          }
        }
        if (matching == False && length>1) {
          penalty += length;
        }
        length=0;
        //printf("matching=%d inDict=%d penalty=%d\n",matching,inDict,penalty);
        matching = False;
      }
    }
  }

  //横
  for (q = 0; q < n; q++) {
    for (p = 0; p < n; p ++) {
      if ((q == 0 && puzzle[p][q] != False) || (q > 0 && puzzle[p][q-1] == False && puzzle[p][q] != False)) {
        for (r=0;r<n-q;r++) {
          if (puzzle[p][q+r] == False) {
            break;
          } else {
            word[r] = puzzle[p][q+r];
            length++;
          }
        }
        /*debug
        for (r=0;r<length;r++) {
          printf("word[%d] = %d ",r,word[r]);
        }
        printf("\n");
        printf("length=%d\n",length);
        */

        //wordが辞書内にあるかをチェック
        for (r=0; r<Dict->m; r++) {
          if (length == Dict->len[r]) {
            for (s=0; s<length; s++) {
              if (word[s] == Dict->x[r][s]) {
                matching = True;
              } else {
                matching = False;
                break;
              }
            }
          }
          if (matching == True) {
            inDict += length;
            break;
          }
        }
        if (matching == False && length>1) {
          penalty += length;
        }
        length=0;
        //printf("matching=%d inDict=%d penalty=%d\n",matching,inDict,penalty);
        matching = False;
      }
    }
  }

  //printf("inDict=%d,penalty=%d\n",inDict,penalty);

  free(word); // <--原口追加 (8/11)

  return inDict - n*penalty;
}

/******************** drop **********************/
//指定された単語を黒白白パターンが崩れない限り削除する
int drop(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int sol_size, int **puzzle, int **enable, int *score, int **cover, int t) {
  int p,q,newScore;
  int length = Dict->len[T[t].k];
  int canDrop = True, remove = True;

  //sol_size=0のときはdrop不可
  if (sol_size == 0) {
    return sol_size;
  }

  //黒白白が崩れる時(カバー配列で2が並んでいる時)はdrop不可
  int tmp = True;
  if (T[t].div == 0) {
    for (p = 0; p < length; p++) {
      if (tmp == False && cover[T[t].i + p][T[t].j] == 2) {
        return sol_size;
      }
      if (cover[T[t].i + p][T[t].j] == 2) {
        tmp = False;
      } else {
        tmp = True;
      }
    }
  } else {
    for (p = 0; p < length; p++) {
      if (tmp == False && cover[T[t].i][T[t].j + p] == 2) {
        return sol_size;
      }
      if (cover[T[t].i][T[t].j + p] == 2) {
        tmp = False;
      } else {
        tmp = True;
      }
    }
  }

  //実際に単語を抜く(coverも変更)
  if (T[t].div == 0) {
    for (p = 0; p < length; p++) {
      cover[T[t].i + p][T[t].j] --;
      if (cover[T[t].i + p][T[t].j] == 0) {
        puzzle[T[t].i + p][T[t].j] = False;
      }
    }
  } else {
    for (p = 0; p < length; p++) {
      cover[T[t].i][T[t].j + p] --;
      if (cover[T[t].i][T[t].j + p] == 0) {
        puzzle[T[t].i][T[t].j + p] = False;
      }
    }
  }

  //scoreの更新
  newScore = calc(n, Dict, puzzle, cover, 0);
  //puzzleの表示
  /*
  printf("--- dropped ---\n");
  printf("[");
  for (p = 0; p < length; p++) {
      if (Dict->en == False) {
        printf("%s", decoder(Dict->inv[Dict->x[T[t].k][p]] + MINCODE_JA));
      } else {
        printf("%c", Dict->inv[Dict->x[T[t].k][p]] + MINCODE_EN);
      }
  }
  printf("] i=%d, j=%d\n",T[t].i,T[t].j);
  */

  //禁止マスの解除
  //縦上
  if (T[t].div == 0 && T[t].i > 0) {
    //上上
    if (T[t].i > 2 && puzzle[T[t].i - 2][T[t].j] != False && puzzle[T[t].i - 3][T[t].j] != False) {
      remove = False;
    }
    //上左
    if (T[t].j > 2 && puzzle[T[t].i - 1][T[t].j - 1] != False && puzzle[T[t].i - 1][T[t].j - 2] != False) {
      remove = False;
    }
    //上右
    if (T[t].j < n - 2 && puzzle[T[t].i - 1][T[t].j + 1] != False && puzzle[T[t].i - 1][T[t].j + 2] != False) {
      remove = False;
    }
    //上解除
    if (remove == True) {
      enable[T[t].i - 1][T[t].j] = True;
    } else {
      remove = True;
    }
  }
  //縦下
  if (T[t].div == 0 && T[t].i + length - 1 < n - 1) {
    //下下
    if (T[t].i + length - 1 > n - 2 && puzzle[T[t].i + length + 1][T[t].j] != False && puzzle[T[t].i + length + 2][T[t].j] != False) {
      remove = False;
    }
    //下左
    if (T[t].j > 2 && puzzle[T[t].i + length][T[t].j - 1] != False && puzzle[T[t].i +length][T[t].j - 2] != False) {
      remove = False;
    }
    //下右
    if (T[t].j < n - 2 && puzzle[T[t].i + length][T[t].j + 1] != False && puzzle[T[t].i + length][T[t].j + 2] != False) {
      remove = False;
    }
    //下解除
    if (remove == True) {
      enable[T[t].i + length][T[t].j] = True;
    }
  }
  //横左
  if (T[t].div == 1 && T[t].j > 0) {
    //左左
    if (T[t].j > 2 && puzzle[T[t].i][T[t].j - 2] != False && puzzle[T[t].i][T[t].j - 3] != False) {
      remove = False;
    }
    //左上
    if (T[t].i > 2 && puzzle[T[t].i - 1][T[t].j - 1] != False && puzzle[T[t].i - 2][T[t].j - 1] != False) {
      remove = False;
    }
    //左下
    if (T[t].i < n - 2 && puzzle[T[t].i + 1][T[t].j - 1] != False && puzzle[T[t].i +2][T[t].j - 1] != False) {
      remove = False;
    }
    //左解除
    if (remove == True) {
      enable[T[t].i][T[t].j-1] = True;
    } else {
      remove = True;
    }
  }
  //横右
  if (T[t].div == 1 && T[t].j + length - 1 < n - 1) {
    //右右
    if (T[t].j + length - 1 < n - 2 && puzzle[T[t].i][T[t].j + length + 1] != False && puzzle[T[t].i][T[t].j + length + 2] != False) {
      remove = False;
    }
    //右上
    if (T[t].i > 2 && puzzle[T[t].i - 1][T[t].j +length] != False && puzzle[T[t].i - 2][T[t].j + length] != False) {
      remove = False;
    }
    //右下
    if (T[t].i < n - 2 && puzzle[T[t].i + 1][T[t].j + length] != False && puzzle[T[t].i + 2][T[t].j + length] != False) {
      remove = False;
    }
    //右解除
    if (remove == True) {
      enable[T[t].i][T[t].j + length] = True;
    }
  }

  //Solの更新
  Sol[t] = False;

  *score = newScore;
  //display(Dict, puzzle, score, sol_size-1, n);
  return sol_size - 1;
}


/*************************check_add*****************************/
//add出来るならTrue,できないならFalseを返す
int check_add(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int sol_size, int t, int **puzzle, int **enable) {
  int p, q, k;
  int crossing = False;
  int length = Dict->len[T[t].k];

  //ひとつも単語が使われてない時はTrue
  if (sol_size == 0) {
    return True;
  }

  //既に使われている場合はFalse
  if (Sol[t] == True) {
    return False;
  }

  //同じ単語が使われている場合はFalse
  for (p = 0; p < t_size; p++) {
    if (Sol[p] == True && T[p].k == T[t].k) {
      return False;
    }
  }

  //単語がパズルに収まらない場合add不可
  if ((T[t].div == 0 && T[t].i + length > n) || (T[t].div == 1 && T[t].j + length > n)) {
    return False;
  }

  //交差を判定する(交差部の文字も判定)
  for (p = 0; p < length; p++) {
    if (T[t].div == 0 && puzzle[T[t].i + p][T[t].j] != False) {
      crossing = True;
      if (puzzle[T[t].i + p][T[t].j] != Dict->x[T[t].k][p]) {
        //printf("puzzle=%d, Dict=%d\n", puzzle[T[t].i + p][T[t].j], Dict->x[T[t].k][p]);
        crossing = False;
        break;
      }
    } else if (T[t].div == 1 && puzzle[T[t].i][T[t].j + p] != False) {
      crossing = True;
      if (puzzle[T[t].i][T[t].j + p] != Dict->x[T[t].k][p]) {
        //printf("puzzle=%d, Dict=%d\n", puzzle[T[t].i][T[t].j + p], Dict->x[T[t].k][p]);
        crossing = False;
        break;
      }
    }
  }
  for (p = 0; p < length; p++) {
    if (crossing == True) {
      for (p = 0; p < length; p++) {
        if (T[t].div == 0) {
          if (enable[T[t].i + p][T[t].j] == False) {
            //printf("! <p=%d (%d,%d) k=%d: %d %d>", p, T[t].i, T[t].j, T[t].k, puzzle[T[t].i+p][T[t].j], Dict->x[T[t].k][p] );
            crossing = False;
            break;
          }
          if (puzzle[T[t].i + p][T[t].j] != False &&
            puzzle[T[t].i + p][T[t].j] != Dict->x[T[t].k][p]) {
            //printf("# <p=%d (%d,%d) k=%d: %d %d>", p, T[t].i, T[t].j, T[t].k, puzzle[T[t].i+p][T[t].j], Dict->x[T[t].k][p] );
            crossing = False;
            break;
          }
        } else {
          if (enable[T[t].i][T[t].j + p] == False) {
            // printf("!! <p=%d (%d,%d) k=%d: %d %d>", p, T[t].i, T[t].j, T[t].k, puzzle[T[t].i][T[t].j+p], Dict->x[T[t].k][p] );
            crossing = False;
            break;
          }
          if (puzzle[T[t].i][T[t].j + p] != False &&
            puzzle[T[t].i][T[t].j + p] != Dict->x[T[t].k][p]) {
            //printf("## <p=%d (%d,%d) k=%d: %d %d>", p, T[t].i, T[t].j, T[t].k, puzzle[T[t].i][T[t].j+p], Dict->x[T[t].k][p] );
            crossing = False;
            break;
          }
        }
      }
    }
  }
  if (crossing == False) {
    return False;
  }

  /*
  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      if (enable[p][q] == False) {
        printf("F ");
      } else {
        printf("T ");
      }
    }
    printf("\n");
    printf("length=%d,i=%d,j=%d\n", length,T[t].i,T[t].j);
  }
  */
  /*
  printf("[");
  for (p = 0; p < length; p++) {
    if (Dict->en == False) {
      printf("%s", decoder(Dict->inv[Dict->x[T[t].k][p]] + MINCODE_JA));
    } else {
      printf("%c", Dict->inv[Dict->x[T[t].k][p]] + MINCODE_EN);
    }
  }
  printf("] i=%d, j=%d, div=%d\n", T[t].i, T[t].j, T[t].div);
  */


  //交差部分以外のマスで隣に文字が入っている場合、add不可（？）
  for (p = 0; p < length; p++) {
    if (T[t].div == 0 && puzzle[T[t].i + p][T[t].j] == False) {
      //左
      if (T[t].j > 0 && puzzle[T[t].i + p][T[t].j - 1] != False) {
        return False;
      }
      //右
      if (T[t].j < n-1 && puzzle[T[t].i + p][T[t].j + 1] != False) {
        return False;
      }
    } else if (T[t].div == 1 && puzzle[T[t].i][T[t].j + p] == False) {
      //上
      if (T[t].i > 0 && puzzle[T[t].i - 1][T[t].j + p] != False) {
        return False;
      }
      //下
      if (T[t].i < n-1 && puzzle[T[t].i + 1][T[t].j + p] != False) {
        return False;
      }
    }
  }

  //単語を配置するマスに禁止マスがある場合add不可
  if (T[t].div == 0) { //縦
    for (p = 0; p < length; p++) {
      if (enable[T[t].i + p][T[t].j] == False) {
        return False;
      }
    }
    //単語の前後に既に文字が入っている場合add不可
    if (T[t].i > 0 && puzzle[T[t].i - 1][T[t].j] != False) {
      return False;
    }
    if (T[t].i + length < n && puzzle[T[t].i + length][T[t].j] != False) {
      return False;
    }
  } else { //横
    for (p = 0; p < length; p++) {
      if (enable[T[t].i][T[t].j + p] == False) {
        return False;
      }
    }
    //単語の前後に既に文字が入っている場合add不可
    if (T[t].j > 0 && puzzle[T[t].i][T[t].j - 1] != False) {
      return False;
    }
    if (T[t].j + length < n && puzzle[T[t].i][T[t].j + length] != False) {
      return False;
    }
  }

  //詰め込み可能ならTrue
  return True;
}

/*************************check_drop*****************************/
//指定された単語がdrop可ならTrue,不可ならFalseを返す
int check_drop(int n, Dictionary *Dict, Tuple *T, int sol_size, int **cover, int t) {
  int p,q;
  int **cover_dfs;
  int connected;
  int canDrop = False;
  int length = Dict->len[T[t].k];

  //sol_size=0のときはdrop不可
  if (sol_size == 0) {
    return False;
  }

  cover_dfs = (int**)malloc(n*sizeof(int*));
  for (p = 0; p < n; p++) {
    cover_dfs[p] = (int*)malloc(n*sizeof(int));
  }

  //黒白白が崩れる時(カバー配列で2が並んでいる時)はdrop不可
  int tmp = True;
  if (T[t].div == 0) {
    for (p = 0; p < length; p++) {
      if (tmp == False && cover[T[t].i + p][T[t].j] == 2) {
        goto END_FALSE;
      }
      if (cover[T[t].i + p][T[t].j] == 2) {
        tmp = False;
      } else {
        tmp = True;
      }
    }
  } else {
    for (p = 0; p < length; p++) {
      if (tmp == False && cover[T[t].i][T[t].j + p] == 2) {
        goto END_FALSE;
      }
      if (cover[T[t].i][T[t].j + p] == 2) {
        tmp = False;
      } else {
        tmp = True;
      }
    }
  }

  //単語を抜いた後のカバー配列
  if (T[t].div == 0) {
    for (p = 0; p < length; p++) {
      cover[T[t].i + p][T[t].j] -= 1;
    }
  } else {
    for (p = 0; p < length; p++) {
      cover[T[t].i][T[t].j + p] -= 1;
    }
  }

  //cover_dfsの作成
  for (p = 0; p<n; p++) {
    for (q = 0; q < n; q++) {
      if (cover[p][q] == 0) {
        cover_dfs[p][q] = -1;
      } else {
        cover_dfs[p][q] = 0;
      }
    }
  }

  //連結性が保たれていればTrue,保たれてなければFalseを返す。（coverは元に戻す）
  connected = getConnectedComponents(n, cover_dfs);

  if (connected == 1) {
    if (T[t].div == 0) {
      for (p = 0; p < length; p++) {
        cover[T[t].i + p][T[t].j] += 1;
      }
    } else {
      for (p = 0; p < length; p++) {
        cover[T[t].i][T[t].j + p] += 1;
      }
    }
    goto END_TRUE;
  } else {
    if (T[t].div == 0) {
      for (p = 0; p < length; p++) {
        cover[T[t].i + p][T[t].j] += 1;
      }
    } else {
      for (p = 0; p < length; p++) {
        cover[T[t].i][T[t].j + p] += 1;
      }
    }
    goto END_FALSE;
  }

END_TRUE:
  for (p = 0; p < n; p++) {
    free(cover_dfs[p]);
  }
  free(cover_dfs);
  return True;

END_FALSE:
  for (p = 0; p < n; p++) {
    free(cover_dfs[p]);
  }
  free(cover_dfs);
  return False;

}

/*************************kick*****************************/
//連結性が崩れている場合、scoreが一番大きい部分だけを残しdropする
void kick(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int *sol_size, int **puzzle, int **enable, int *score, int **cover) {

  if (*sol_size == 0) {
    return;
  }

  int p, q,r, t, connected, num=0;
  int **cover_dfs;

  cover_dfs = (int**)malloc(n * sizeof(int*));
  for (p = 0; p < n; p++) {
    cover_dfs[p] = (int*)malloc(n * sizeof(int));
  }

  //cover_dfsを作成
  for (p = 0; p<n; p++) {
    for (q = 0; q < n; q++) {
      if (cover[p][q] == 0) {
        cover_dfs[p][q] = -1;
      } else {
        cover_dfs[p][q] = 0;
      }
    }
  }

  //連結成分の数
  connected = getConnectedComponents(n, cover_dfs);

  /*//cover_dfsを表示
  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      printf("%d ", cover_dfs[p][q]);
    }
    printf("\n");
  }*/

  int *arr_score;
  arr_score = (int*)calloc((n+1)/2 , sizeof(int*));

  //連結成分ごとのスコアを計算
  if (connected != 1) {
    for (p = 0; p < n; p++) {
      for (q = 0; q < n; q++) {
        if (cover_dfs[p][q] > 0) {
          arr_score[cover_dfs[p][q] - 1] += cover[p][q];
        }
      }
    }

    //スコアが一番大きい成分の番号を決定
    for (p = 0; p < (n + 1) / 2; p++) {
      if (arr_score[p] > arr_score[num]) {
        num = p;
      }
    }

    //printf("--- start kick ---\n");

    //num部分をすべてdrop

    while (1) {
      r = *sol_size;
      for (t = 0; t < t_size; t++) {
        if (Sol[t] == True) {
          if (cover_dfs[T[t].i][T[t].j] > 0 && cover_dfs[T[t].i][T[t].j] != num + 1) {
            *sol_size = drop(n, Dict, T, t_size, Sol, *sol_size, puzzle, enable, score, cover, t);
          }
        }
      }
      if (r == *sol_size) {
        break;
      }
    }
  }

  for (p = 0; p < n; p++) {
    free(cover_dfs[p]);
  }
  free(cover_dfs);
  free(arr_score);
}
/*************配列をシャッフルする**************/
void shuffle(int arr[], int size) {
  for (int i = 0; i<size; i++) {
    int j = rand() % size;
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
  }
}

/*************パズルを表示する**************/
void display(Dictionary *Dict, int **puzzle, int *score, int sol_size, int n) {
  int p, q;
  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      if (puzzle[p][q] >= 0) {
        if (Dict->en == False) {
          printf("%s", decoder(Dict->inv[puzzle[p][q]] + MINCODE_JA));
        } else {
          printf("%c ", Dict->inv[puzzle[p][q]] + MINCODE_EN);
        }
      } else {
        if (Dict->en == False) {
          printf("□");
        } else {
          printf("- ");
        }
      }
    }
    printf("\n");
  }
  printf("score:%d sol_size:%d\n\n", *score, sol_size);
}

/****************連結成分を移動できる方向の内ランダムに平行移動する*****************/

int move(int n, Dictionary *Dict,
    Tuple *T, int t_size, int *Sol, int ****InvT,
    int **puzzle, int *score, int sol_size, int **enable, int **cover) {

  if (sol_size == 0) {
    return False;
  }

  int p, q, r, move;
  int direction[4] = {0, 1, 2, 3}; //0:上 1:下 2:左 3:右
  int a = False, t,s,div,i,j,k,*I;

  //方向を決める
  shuffle(direction, 4);

  move = check_move(n, puzzle, enable, cover, direction[0]);
  if (move == 0) {
    return False;
  }

  a = direction[p];

  /*
  printf("Before\n");

  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      if (enable[p][q] == False) {
        printf("F ");
      } else {
        printf("T ");
      }
    }
    printf("\n");
  }
  printf("\n");
  */

  //平行移動する
  if (a != False) {
    //printf("--- moved (a:%d, move:%d)---\n", a, move);
    switch (a) {
    case 0://上
      //移動
      for (p = 0; p < n - move; p++) {
        for (q = 0; q < n; q++) {
          puzzle[p][q] = puzzle[p + move][q];
          cover[p][q] = cover[p + move][q];
          enable[p][q] = enable[p + move][q];
        }
      }
      //削除
      for (p = n - 1; p > n - move - 1; p--) {
        for (q = 0; q < n; q++) {
          puzzle[p][q] = False;
          cover[p][q] = 0;
          enable[p][q] = True;
        }
      }
      //禁止マス再設定
      if (move != n - 1) {
        for (p = 0; p < n; p++) {
          if (puzzle[n - move - 1][p] != False && puzzle[n - move - 2][p] != False) {
            enable[n - move][p] = False;
          }
        }
      }

      break;

    case 1://下
      //移動
      for (p = n - 1; p >= move; p--) {
        for (q = 0; q < n; q++) {
          puzzle[p][q] = puzzle[p - move][q];
          cover[p][q] = cover[p - move][q];
          enable[p][q] = enable[p - move][q];
        }
      }
      //削除
      for (p = 0; p < move; p++) {
        for (q = 0; q < n; q++) {
          puzzle[p][q] = False;
          cover[p][q] = 0;
          enable[p][q] = True;
        }
      }
      //禁止マス再設定
      if (move != n - 1) {
        for (p = 0; p < n; p++) {
          if (puzzle[move][p] != False && puzzle[move + 1][p] != False) {
            enable[move - 1][p] = False;
          }
        }
      }
      break;
    case 2://左
      //移動
      for (p = 0; p < n - move; p++) {
        for (q = 0; q < n; q++) {
          puzzle[q][p] = puzzle[q][p + move];
          cover[q][p] = cover[q][p + move];
          enable[q][p] = enable[q][p + move];
        }
      }
      //削除
      for (p = n - 1; p > n - move - 1; p--) {
        for (q = 0; q < n; q++) {
          puzzle[q][p] = False;
          cover[q][p] = 0;
          enable[q][p] = True;
        }
      }
      //禁止マス再設定
      if (move != n - 1) {
        for (p = 0; p < n; p++) {
          if (puzzle[p][n - move - 1] != False && puzzle[p][n - move - 2] != False) {
            enable[p][n - move] = False;
          }
        }
      }
      break;
    case 3://右
      //移動
      for (p = n - 1; p >= move; p--) {
        for (q = 0; q < n; q++) {
          puzzle[q][p] = puzzle[q][p - move];
          cover[q][p] = cover[q][p - move];
          enable[q][p] = enable[q][p - move];
        }
      }
      //削除
      for (p = 0; p < move; p++) {
        for (q = 0; q < n; q++) {
          puzzle[q][p] = False;
          cover[q][p] = 0;
          enable[q][p] = True;
        }
      }
      //禁止マス再設定
      if (move != n - 1) {
        for (p = 0; p < n; p++) {
          if (puzzle[p][move] != False && puzzle[p][move + 1] != False) {
            enable[p][move - 1] = False;
          }
        }
      }
      break;
    default:
      printf("error\n");
    }
    /*
    printf("After\n");

    for (p = 0; p < n; p++) {
      for (q = 0; q < n; q++) {
        if (enable[p][q] == False) {
          printf("F ");
        } else {
          printf("T ");
        }
      }
      printf("\n");
    }
    printf("\n");
    */

    // Solを更新
    I = (int*)malloc(sol_size*sizeof(int));
    s = 0;
    for (t=0;t<t_size;t++) {
      if (Sol[t]==True) {
        I[s] = t;
        s++;
      }
    }
    if (s!=sol_size) {
      fprintf(stderr,"error: something is strange.\n");
      exit(1);
    }

    //printf("  update Sol...(t_size=%d, a=%d, move=%d)\n", t_size, a, move);
    for (p=0;p<sol_size;p++) {
      t = I[p];

      //printf("  t=%d  ",t); fflush(stdout);

      Sol[t] = False;
      div = T[t].div;
      k = T[t].k;
      i = T[t].i;
      j = T[t].j;

      //printf("(%d,%d,%d,%d)  ",div,k,i,j); fflush(stdout);

      switch (a) {
      case 0: //上
        i = i-move;
        break;
      case 1: //下
        i = i+move;
        break;
      case 2: //左
        j = j-move;
        break;
      case 3: //右
        j = j+move;
        break;
      }

      //printf("---> (%d,%d,%d,%d)  ",div,k,i,j); fflush(stdout);

      s = InvT[div][k][i][j];

      //printf("s=%d\n",s);

      Sol[s] = True;
    }
    //printf("  done.\n\n");
  }


  //display(Dict, puzzle, score, sol_size, n);

  return True;
}

/***********指定された方向に何マス平行移動できるかを返す関数************/
int check_move(int n, int **puzzle, int **enable, int **cover, int direction) {
  int p, q, move=0, canMove = False;

  switch (direction) {
  case 0:
    for (p = 0; p < n; p++) {
      for (q = 0; q < n; q++) {
        if (cover[p][q] == 0) {
          canMove = True;
        } else {
          canMove = False;
          break;
        }
      }
      if (canMove == True) {
        move++;
      } else {
        break;
      }
    }
    break;

  case 1:
    for (p = n-1; p >= 0; p--) {
      for (q = 0; q < n; q++) {
        if (cover[p][q] == 0) {
          canMove = True;
        } else {
          canMove = False;
          break;
        }
      }
      if (canMove == True) {
        move++;
      } else {
        break;
      }
    }
    break;
  case 2:
    for (p = 0; p < n; p++) {
      for (q = 0; q < n; q++) {
        if (cover[q][p] == 0) {
          canMove = True;
        } else {
          canMove = False;
          break;
        }
      }
      if (canMove == True) {
        move++;
      } else {
        break;
      }
    }
    break;
  case 3:
    for (p = n - 1; p >= 0; p--) {
      for (q = 0; q < n; q++) {
        if (cover[q][p] == 0) {
          canMove = True;
        } else {
          canMove = False;
          break;
        }
      }
      if (canMove == True) {
        move++;
      } else {
        break;
      }
    }
    break;
  default:
    printf("error!\n");
  }

  return move;
}

/*******利益計算*******/
int calc_profit(Dictionary *Dict, Tuple *T, int t_size, int *Sol) {
  int p, profit=0;
  for (p = 0; p < t_size; p++) {
    if (Sol[p] == True) {
      profit += Dict->p[T[p].k];
    }
  }
  return profit;
}

/********連結性が崩れるまでdrop*******/
void breakConnection(int n, Dictionary *Dict, Tuple *T, int t_size, int *Sol, int *sol_size, int t, int **puzzle, int **enable, int *score, int **cover) {
  int p, q, r, count, *arr_drop, **cover_dfs;

  //連結性確認用配列
  cover_dfs = (int**)malloc(n * sizeof(int*));
  for (p = 0; p < n; p++) {
    cover_dfs[p] = (int*)malloc(n * sizeof(int));
  }

  //ランダム順にdropするための配列
  arr_drop = (int*)malloc(*sol_size * sizeof(int));
  for (p = 0; p < *sol_size; p++) {
    arr_drop[p] = p;
  }

  shuffle(arr_drop, *sol_size);

  //連結性が崩れるまでdrop
  for (p = 0; p < *sol_size; p++) {
    //1つdrop
    count = 0;
    for (q = 0; q < t_size; q++) {
      if (Sol[q] == True) {
        //printf("p=%d, q=%d, Sol[q]=%d, arr_drop[p]=%d\n", p, q, Sol[q], arr_drop[p]);
        if (count == arr_drop[p]) {
          *sol_size = drop(n, Dict, T, t_size, Sol, *sol_size, puzzle, enable, score, cover, q);
          break;
        }
        count++;
      }
    }

    //cover_dfsを作成
    for (q = 0; q < n; q++) {
      for (r = 0; r < n; r++) {
        if (cover[q][r] == 0) {
          cover_dfs[q][r] = -1;
        } else {
          cover_dfs[q][r] = 0;
        }
      }
    }

    //連結生が崩れたら終了
    if (getConnectedComponents(n, cover_dfs) > 1) {
      //printf("--- broke connection ---\n\n");
      break;
    }
  }

  //メモリ解放
  for (p = 0; p < n; p++) {
    free(cover_dfs[p]);
  }
  free(cover_dfs);
  free(arr_drop);
}
