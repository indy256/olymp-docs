#include <cstdio>
#include <algorithm>

using namespace std;

#define DEPTH 10

#define X_DEPTH DEPTH
#define Y_DEPTH DEPTH

#define MAX(x,y) ((x)>(y)?(x):(y))

#define X_SIZE (1<<(X_DEPTH+1))
#define Y_SIZE (1<<(Y_DEPTH+1))

typedef long long ll;

//Массив логарифмов, а также инициализируящая его функция
int log2d[ MAX(X_SIZE, Y_SIZE) ];
void init_log2d(){
   int d = 0;
   int i = 0;
   while(d<=MAX(X_DEPTH, Y_DEPTH)){
      if(i==(1<<(d+1))){
         d++;
      }
      log2d[i] = d;
      i++;
   }
}

//Количество ячеек в отрезке текущей вершины
int ncount(int n, int depth){
   return 1<<(depth-log2d[n]);
}

//Индекс левой границы отрезка текущей вершины
int first_index(int n, int count){
   return (n - (1<<log2d[n]))*count;
}

//Функция, собирающая списки вершин для обхода
void fill_node_set(int l, int r, int depth, int* arr, int & arr_count, int node = 1, int nl = 0, int nr = 0){
   arr[arr_count++] = node;
   if(node==1) nr = (1<<depth)-1;
   if(nl==l&&nr==r) return;
   int ll = nl, lr = ((nl+nr)>>1);
   int rl = lr+1, rr=nr;
   if(rl<=l) fill_node_set(l,r,depth,arr,arr_count,(node<<1)+1,rl,rr);
   else if(lr>=r) fill_node_set(l,r,depth,arr,arr_count,(node<<1),ll,lr);
   else{
      fill_node_set(l,lr,depth,arr,arr_count,(node<<1),ll,lr);
      fill_node_set(rl,r,depth,arr,arr_count,(node<<1)+1,rl,rr);
   }
}

//Списки вершин и количества элементов в них
int x_set[100], y_set[100], x_count = 0, y_count = 0;

//Количество элементов в пересечении отрезков, для вычисления xc и yc
int intersect_count(int nl, int nr, int l, int r){
   if(nl<=l&&r<=nr) return r-l+1;
   if(l<=nl&&nr<=r) return nr-nl+1;
   if(nr<l||r<nl) return 0;
   if(l<=nl) return r-nl+1;
   /*if(nl<l)*/ return nr-l+1;
}

//Собственно, двумерное дерево отрезков
struct segtree2d{
   ll add_x[X_SIZE][Y_SIZE], add_y[X_SIZE][Y_SIZE], value[X_SIZE][Y_SIZE], add[X_SIZE][Y_SIZE];
   
   //Операция модификации
   void modify(int x1, int x2, int y1, int y2, int val){
      x_count = 0, y_count = 0;
      fill_node_set(x1,x2,X_DEPTH,x_set,x_count);
      fill_node_set(y1,y2,Y_DEPTH,y_set,y_count);
      
      for(int xi=0; xi<x_count; xi++){
         int _xc = ncount(x_set[xi], X_DEPTH);
         int xl = first_index(x_set[xi], _xc);
         int xr = xl+_xc-1;
         int xc = intersect_count(xl, xr, x1, x2);
         for(int yi=0; yi<y_count; yi++){
            int _yc = ncount(y_set[yi], Y_DEPTH);
            int yl = first_index(y_set[yi], _yc);
            int yr = yl+_yc-1;
            int yc = intersect_count(yl, yr, y1, y2);
            
            bool b1 = y1<=yl&&yr<=y2,
                 b2 = x1<=xl&&xr<=x2;
            if(b1&&b2) add[x_set[xi]][y_set[yi]]+=val;
            else{
               if(b1) add_x[x_set[xi]][y_set[yi]]+=val*xc;
               if(b2) add_y[x_set[xi]][y_set[yi]]+=val*yc;
            }
            if(!b1&&!b2){
               value[x_set[xi]][y_set[yi]] += ((ll)xc)*yc*val;
            }
         }
      }
   }
   
   //Операция суммы
   ll summ(int x1, int x2, int y1, int y2){
      x_count = 0, y_count = 0;
      ll res = 0;
      fill_node_set(x1,x2,X_DEPTH,x_set,x_count);
      fill_node_set(y1,y2,Y_DEPTH,y_set,y_count);
      
      for(int xi=0; xi<x_count; xi++){
         int _xc = ncount(x_set[xi], X_DEPTH);
         int xl = first_index(x_set[xi], _xc);
         int xr = xl+_xc-1;
         int xc = intersect_count(xl, xr, x1, x2);
         for(int yi=0; yi<y_count; yi++){
            int _yc = ncount(y_set[yi], Y_DEPTH);
            int yl = first_index(y_set[yi], _yc);
            int yr = yl+_yc-1;
            int yc = intersect_count(yl, yr, y1, y2);
            
            int xvl = max(xl,x1), xvr = min(xr,x2), yvl = max(yl,y1), yvr = min(yr,y2);
            
            res+=yc*xc*add[x_set[xi]][y_set[yi]];
            
            if(xl==xvl&&xvr==xr) res+=yc*add_x[x_set[xi]][y_set[yi]];
            if(yl==yvl&&yvr==yr) res+=xc*add_y[x_set[xi]][y_set[yi]];
            
            if(x1<=xl&&xr<=x2&&y1<=yl&&yr<=y2) res+=value[x_set[xi]][y_set[yi]];
         }
      }
      return res;
   }
   
} tree;


int main(int argc, char** argv) {
	freopen("segtree2d.in","r",stdin);
	freopen("segtree2d.out","w",stdout);
   init_log2d();
   int __x,__y,N,x1,y1, x2, y2, op, w;
   scanf("%d%d%d", &__x, &__y, &N);
   for(int i=0; i<N; i++){
      scanf("%d%d%d%d%d", &op, &x1, &y1, &x2, &y2);
      x1--, y1--, x2--, y2--;
      if(op==1){
         scanf("%d", &w);
         tree.modify(x1,x2,y1,y2,w);
      }else{
         printf("%lld\n", tree.summ(x1,x2,y1,y2));
      }
   }
   
	return 0;
}
