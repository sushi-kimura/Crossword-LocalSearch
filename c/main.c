/******************** main.c ********************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

#include "basics.h"
#include "common.h"

#define True -1
#define False -2

int main(int argc, char *argv[]) {
  int div;        // 縦横の添え字
  int i;          // 行の添え字
  int j;          // 列の添え字
  int k;          // 単語の添え字
  int d;          // 単語における文字の添え字
  int t = 0;        // タプル集合におけるタプルの添え字
  int code;       // 文字コード
  int ***InvT[2];   // Tの逆写像 InvT[div][k][i][j] は該当するタプルの T における添字
  int *Sol;       // 解であるタプルの添字集合
  int t_size;     // T に属するタプルの個数
  int sol_size = 0; // 解 Sol に属するタプルの個数
  int profit = 0;     //重みの総和
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
  cover = (int**)malloc(n * sizeof(int*));
  for (p = 0; p < n; p++) {
    cover[p] = (int*)malloc(n * sizeof(int));
  }
  for (p = 0; p < n; p++) {
      for (q = 0; q < n; q++) {
          cover[p][q] = 0;
      }
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
      sol_size = add(t_size, Sol, sol_size, arr_rand[t], puzzle, enable, &score, cover);
    }
  }
  profit = calc_profit(t_size, Sol);

  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      if (cover[p][q] != 0) {
        white++;
      }
    }
  }

  clock_t now = clock();//現在時間
  display(puzzle, &score, sol_size, n);
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
    breakConnection(t_size, Sol, &sol_size, t, puzzle, enable, &score, cover);

    //kickする
    kick(t_size, Sol, &sol_size, puzzle, enable, &score, cover);

    //移動できる方向をランダムで選び、平行移動する
    if (movetype == 1) {
      if (move(t_size, Sol, InvT, puzzle, &score, sol_size, enable, cover) == True) {
        moveCount++;
      }
    }

    //詰められるだけadd
    shuffle(arr_rand,t_size);
    tmp = -1;
    while (tmp != sol_size) {
      tmp = sol_size;
      for (t = 0; t < t_size; t++) {
        sol_size = add(t_size, Sol, sol_size, arr_rand[t], puzzle, enable, &score, cover);
      }
    }

    profit = calc_profit(t_size, Sol);

    white = 0;
    for (p = 0; p < n; p++) {
      for (q = 0; q < n; q++) {
        if (cover[p][q] != 0) {
          white++;
        }
      }
    }

    //改善していれば繰り返す
    //if ((white > tmp_white) || (white == tmp_white && score >= tmp_score)) {
    int tmp_black_count = black_connection_max(tmp_cover);
    int black_count = black_connection_max(cover);
    if (black_count < tmp_black_count || (black_count == tmp_black_count && black_max_count(cover, black_count) < black_max_count(tmp_cover, black_count))) {
      changed = False;
      //改善解の表示
      improveCount++;
      printf("--- improved[%d] ---\n\n", improveCount);
      display(puzzle, &score, sol_size, n);
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
  display(puzzle, &score, sol_size, n);
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
