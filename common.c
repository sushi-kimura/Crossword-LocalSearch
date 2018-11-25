/******************** common.c ********************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "basics.h"
#include "common.h"

/******************** add **********************/
int add(int t_size, int *Sol, int sol_size, int t, int **puzzle, int **enable, int *score, int **cover) {
  int p, q, k;
  //int *haitteta;
  int canAdd;
  int length = Dict.len[T[t].k];

  //単語が詰め込み可能であるかどうか
  canAdd = check_add(t_size, Sol, sol_size, t, puzzle, enable);
  if (canAdd == False) {
    return sol_size;
  }

  //盤面に配置
 // haitteta = (int*)malloc(length*sizeof(int));
  for (p = 0; p < length; p++) {
    if (T[t].div == 0) {
      //haitteta[p] = puzzle[T[t].i+p][T[t].j];
      puzzle[T[t].i + p][T[t].j] = Dict.x[T[t].k][p];
      //printf("<p=%d, %d>",p,Dict.x[T[t].k][p]);
    } else {
      //haitteta[p] = puzzle[T[t].i][T[t].j + p];
      puzzle[T[t].i][T[t].j + p] = Dict.x[T[t].k][p];
    }
  }
  /*
  //解が改善しなかった場合は終了
  int newScore = calc(puzzle, cover, 0);
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
  *score = calc(puzzle, cover, 0);

  //puzzleの表示
  //printf("--- added ---\n\n");
  //display(puzzle, score, sol_size+1, n);

  return sol_size+1;

END:
  //free(haitteta);
  return sol_size;
}


/******************** calc **********************/
//mode=0のときはcover配列上の値の合計を返す。
int calc(int **puzzle, int **cover, int mode) {
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
  int matching=False;
  int inDict=0; //パズルにある単語で辞書内にある単語と一致する数
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
        for (r=0; r<Dict.m; r++) {
          if (length == Dict.len[r]) {
            for (s=0; s<length; s++) {
              if (word[s] == Dict.x[r][s]) {
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
        for (r=0; r<Dict.m; r++) {
          if (length == Dict.len[r]) {
            for (s=0; s<length; s++) {
              if (word[s] == Dict.x[r][s]) {
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
int drop(int t_size, int *Sol, int sol_size, int **puzzle, int **enable, int *score, int **cover, int t) {
  int p,q,newScore;
  int length = Dict.len[T[t].k];
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
  newScore = calc(puzzle, cover, 0);
  //puzzleの表示
  /*
  printf("--- dropped ---\n");
  printf("[");
  for (p = 0; p < length; p++) {
      if (Dict.en == False) {
        printf("%s", decoder(Dict.inv[Dict.x[T[t].k][p]] + MINCODE_JA));
      } else {
        printf("%c", Dict.inv[Dict.x[T[t].k][p]] + MINCODE_EN);
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
  //display(score, sol_size-1, n);
  return sol_size - 1;
}


/*************************check_add*****************************/
//add出来るならTrue,できないならFalseを返す
int check_add(int t_size, int *Sol, int sol_size, int t, int **puzzle, int **enable) {
  int p, q, k;
  int crossing = False;
  int length = Dict.len[T[t].k];

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
      if (puzzle[T[t].i + p][T[t].j] != Dict.x[T[t].k][p]) {
        //printf("puzzle=%d, Dict=%d\n", puzzle[T[t].i + p][T[t].j], Dict.x[T[t].k][p]);
        crossing = False;
        break;
      }
    } else if (T[t].div == 1 && puzzle[T[t].i][T[t].j + p] != False) {
      crossing = True;
      if (puzzle[T[t].i][T[t].j + p] != Dict.x[T[t].k][p]) {
        //printf("puzzle=%d, Dict=%d\n", puzzle[T[t].i][T[t].j + p], Dict.x[T[t].k][p]);
        crossing = False;
        break;
      }
    }
  }

  if (crossing == True) {
      for (p = 0; p < length; p++) {
        if (T[t].div == 0) { //縦
          if (enable[T[t].i+p][T[t].j] == False) {
            //printf("! <p=%d (%d,%d) k=%d: %d %d>", p, T[t].i, T[t].j, T[t].k, puzzle[T[t].i+p][T[t].j], Dict.x[T[t].k][p] );
            crossing = False;
            break;
          }
        } else { //横
          if (enable[T[t].i][T[t].j+p] == False) {
            // printf("!! <p=%d (%d,%d) k=%d: %d %d>", p, T[t].i, T[t].j, T[t].k, puzzle[T[t].i][T[t].j+p], Dict.x[T[t].k][p] );
            crossing = False;
            break;
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
    if (Dict.en == False) {
      printf("%s", decoder(Dict.inv[Dict.x[T[t].k][p]] + MINCODE_JA));
    } else {
      printf("%c", Dict.inv[Dict.x[T[t].k][p]] + MINCODE_EN);
    }
  }
  printf("] i=%d, j=%d, div=%d\n", T[t].i, T[t].j, T[t].div);
  */


  //交差部分以外のマスで隣に文字が入っている場合、add不可（？）
  for (p = 0; p < length; p++) {
    if (T[t].div == 0 && puzzle[T[t].i+p][T[t].j] == False) {
      //左
      if (T[t].j > 0 && puzzle[T[t].i+p][T[t].j-1] != False) {
        return False;
      }
      //右
      if (T[t].j < n-1 && puzzle[T[t].i+p][T[t].j+1] != False) {
        return False;
      }
    } else if (T[t].div == 1 && puzzle[T[t].i][T[t].j+p] == False) {
      //上
      if (T[t].i > 0 && puzzle[T[t].i-1][T[t].j+p] != False) {
        return False;
      }
      //下
      if (T[t].i < n-1 && puzzle[T[t].i+1][T[t].j+p] != False) {
        return False;
      }
    }
  }

  //単語を配置するマスに禁止マスがある場合add不可
  if (T[t].div == 0) { //縦
    for (p = 0; p < length; p++) {
      if (enable[T[t].i+p][T[t].j] == False) {
        return False;
      }
    }
    //単語の前後に既に文字が入っている場合add不可
    if (T[t].i > 0 && puzzle[T[t].i-1][T[t].j] != False) {
      return False;
    }
    if (T[t].i+length < n && puzzle[T[t].i + length][T[t].j] != False) {
      return False;
    }
  } else { //横
    for (p = 0; p < length; p++) {
      if (enable[T[t].i][T[t].j+p] == False) {
        return False;
      }
    }
    //単語の前後に既に文字が入っている場合add不可
    if (T[t].j > 0 && puzzle[T[t].i][T[t].j-1] != False) {
      return False;
    }
    if (T[t].j+length < n && puzzle[T[t].i][T[t].j + length] != False) {
      return False;
    }
  }

  //詰め込み可能ならTrue
  return True;
}

/*************************check_drop*****************************/
//指定された単語がdrop可ならTrue,不可ならFalseを返す
int check_drop(int sol_size, int **cover, int t) {
  int p,q;
  int **cover_dfs;
  int connected;
  int canDrop = False;
  int length = Dict.len[T[t].k];

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
void kick(int t_size, int *Sol, int *sol_size, int **puzzle, int **enable, int *score, int **cover) {

  if (*sol_size == 0) {
    return;
  }

  int p, q, r, t, connected, num=0;
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
            *sol_size = drop(t_size, Sol, *sol_size, puzzle, enable, score, cover, t);
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
void display(int **puzzle, int *score, int sol_size, int n) {
  int p, q;
  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      if (puzzle[p][q] >= 0) {
        if (Dict.en == False) {
          printf("%s", decoder(Dict.inv[puzzle[p][q]] + MINCODE_JA));
        } else {
          printf("%c ", Dict.inv[puzzle[p][q]] + MINCODE_EN);
        }
      } else {
        if (Dict.en == False) {
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
int move(int t_size, int *Sol, int ****InvT, int **puzzle, int *score, int sol_size, int **enable, int **cover) {

  if (sol_size == 0) {
    return False;
  }

  int p, q, r, move;
  int direction[4] = {0, 1, 2, 3}; //0:上 1:下 2:左 3:右
  int a = False, t,s,div,i,j,k,*I;

  //方向を決める
  shuffle(direction, 4);

  move = check_move(puzzle, enable, cover, direction[0]);
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


  //display(puzzle, score, sol_size, n);

  return True;
}

/***********指定された方向に何マス平行移動できるかを返す関数************/
int check_move(int **puzzle, int **enable, int **cover, int direction) {
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
int calc_profit(int t_size, int *Sol) {
  int p, profit=0;
  for (p = 0; p < t_size; p++) {
    if (Sol[p] == True) {
      profit += Dict.p[T[p].k];
    }
  }
  return profit;
}

/********連結性が崩れるまでdrop*******/
void breakConnection(int t_size, int *Sol, int *sol_size, int t, int **puzzle, int **enable, int *score, int **cover) {
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
          *sol_size = drop(t_size, Sol, *sol_size, puzzle, enable, score, cover, q);
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

/*************************黒マスの最大連結数を返す*****************************/
int black_connection_max(int **cover) {

  int p, q, connected, max=0;
  int **cover_dfs;

  cover_dfs = (int**)malloc(n * sizeof(int*));
  for (p = 0; p < n; p++) {
    cover_dfs[p] = (int*)malloc(n * sizeof(int));
  }

  //cover_dfsを作成
  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      if (cover[p][q] == 0) {
        cover_dfs[p][q] = 0;
      } else {
        cover_dfs[p][q] = -1;
      }
    }
  }

  //連結成分に採番
  connected = getConnectedComponents(n, cover_dfs);

  int *arr_score;
  arr_score = (int*)calloc(n*n/2 , sizeof(int*));

  //連結成分ごとのスコアを計算
  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      if (cover_dfs[p][q] > 0) {
        arr_score[cover_dfs[p][q] - 1] ++;
      }
    }
  }

  //スコアが一番大きい成分の番号を決定
  for (p = 0; p < n*n/2; p++) {
    if (arr_score[p] > max) {
      max = arr_score[p];
    }
  }

  for (p = 0; p < n; p++) {
    free(cover_dfs[p]);
  }
  free(cover_dfs);
  free(arr_score);

  return max;
}

/*************************指定された黒マス連結成分の総数を返す*****************************/
int black_max_count(int **cover, int connection_count) {

  int p, q, connected, count=0;
  int **cover_dfs;

  cover_dfs = (int**)malloc(n * sizeof(int*));
  for (p = 0; p < n; p++) {
    cover_dfs[p] = (int*)malloc(n * sizeof(int));
  }

  //cover_dfsを作成
  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      if (cover[p][q] == 0) {
        cover_dfs[p][q] = 0;
      } else {
        cover_dfs[p][q] = -1;
      }
    }
  }

  //連結成分に採番
  connected = getConnectedComponents(n, cover_dfs);

  int *arr_score;
  arr_score = (int*)calloc(n*n/2 , sizeof(int*));

  //連結成分ごとのスコアを計算
  for (p = 0; p < n; p++) {
    for (q = 0; q < n; q++) {
      if (cover_dfs[p][q] > 0) {
        arr_score[cover_dfs[p][q] - 1] ++;
      }
    }
  }

  //スコアが一番大きい成分の番号を決定
  for (p = 0; p < n*n/2; p++) {
    if (arr_score[p] == connection_count) {
      count++;
    }
  }

  for (p = 0; p < n; p++) {
    free(cover_dfs[p]);
  }
  free(cover_dfs);
  free(arr_score);
  return count;
}
