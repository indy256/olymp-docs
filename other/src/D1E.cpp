#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
//#include <ctime>
#include <map>
using namespace std;

int n, d;
int cost[4001][4001];
int dp[4001][4001];

int totalCost(int L, int R)
{
	return cost[R][R] - cost[R][L-1] - cost[L-1][R] + cost[L-1][L-1];
}

int calc(int divs, int pos, int searchL, int searchR)
{
	dp[divs][pos] = 1000000000;
	int ret = searchL;
	for(int i = searchL; i <= searchR; i++)
	{
		int t = dp[divs-1][i] + totalCost(i+1, pos);
		if(t < dp[divs][pos])
		{
			dp[divs][pos] = t;
			ret = i;
		}
	}
	return ret;
}

void solve(int divs, int L, int R, int searchL, int searchR)
{
	if(L > R)
		return;
	if(L == R)
	{
		calc(divs, L, searchL, searchR);
		return;
	}
	searchR = min(searchR, R-1);
	if(searchL == searchR)
	{
		for(int i = L; i <= R; i++)
			calc(divs, i, searchL, searchR);
		return;
	}
	int M = (L + R) / 2;
	int optM = calc(divs, M, searchL, searchR);
	solve(divs, L, M-1, searchL, optM);
	solve(divs, M+1, R, optM, searchR);
}

char buff[8001];

int MAIN()
{
	scanf("%d %d\n", &n, &d);
	memset(cost, 0, sizeof(cost));
	for(int i = 1; i <= n; i++)
	{
		gets(buff);
		for(int j = 1; j <= n; j++)
		{
			cost[i][j] = buff[j*2-2] - '0';
		}
	}
	for(int i = 1; i <= n; i++)
		for(int j = 1; j <= n; j++)
			cost[i][j] += cost[i-1][j] + cost[i][j-1] - cost[i-1][j-1];

	for(int i = 1; i <= n; i++)
		dp[1][i] = totalCost(1, i);
	for(int i = 2; i <= d; i++)
		solve(i, 2, n, i-1, n);
	
	cout << dp[d][n] / 2 << endl;
	return 0;
}

int main()
{
	//srand((unsigned)time(NULL));
	#ifdef LOCAL_TEST
		freopen("in.txt", "r", stdin);
		freopen("out.txt", "w", stdout);
	#endif
	//ios :: sync_with_stdio(false);
	//cout << fixed << setprecision(2);
	return MAIN();
}
