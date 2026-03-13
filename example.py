class Solution(object):
    def minOperations(self, s):
        ile1 = 0
        ile0 = 0
        lists = list(s)
        for i in range(len(s)):
            if i % 2 == 0:
                if s[i] == '1':
                    ile0 += 1
                else:
                    ile1 += 1
            else:
                if s[i] == '1':
                    ile1 += 1
                else:
                    ile0 += 1
        if ile1 < ile0: ile0 = ile1
        return ile0

    def myAtoi(self, s):
        anslist = list(s)
        ans = 0
        isnegative = False
        wasdigit = False
        wasother = False
        if len(s) == 0: return 0
        while anslist[0] == '0' or anslist[0] == ' ' or anslist[0] == '-' or anslist[0] == '+':
            if wasother or wasdigit: break
            if anslist[0] == '0':
                wasdigit = True
            if anslist[0] == '-':
                if wasdigit: break
                isnegative = True
                wasother = True
            if anslist[0] == '+':
                wasother = True
            anslist.pop(0)
            if len(anslist) == 0: return 0
        for i in range(len(anslist)):
            if not anslist[i].isdigit(): break
            ans *= 10
            ans += int(anslist[i])
        if isnegative: ans = -ans
        if ans > 2 ** 31 - 1: ans = 2 ** 31 - 1
        if ans < -2 ** 31: ans = -2 ** 31
        return ans
if __name__ == '__main__':
    s = Solution()
    print(s.minOperations('1010001001111010110'))