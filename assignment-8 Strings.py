

                    #Answer -------- 1


def minimumDeleteSum(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Fill in the first row and column
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + ord(s1[i-1])
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + ord(s2[j-1])

    # Fill in the rest of the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + ord(s1[i-1]), dp[i][j-1] + ord(s2[j-1]))

    return dp[n][m]


s1 = "sea"
s2 = "eat"
print(minimumDeleteSum(s1, s2))  # Output: 231








                  #Answer -------- 2





def checkValidString(s):
    left_parentheses = []
    asterisks = []

    for i, ch in enumerate(s):
        if ch == '(':
            left_parentheses.append(i)
        elif ch == '*':
            asterisks.append(i)
        else:  # ch == ')'
            if left_parentheses:
                left_parentheses.pop()
            elif asterisks:
                asterisks.pop()
            else:
                return False

    while left_parentheses and asterisks:
        if left_parentheses[-1] > asterisks[-1]:
            return False
        left_parentheses.pop()
        asterisks.pop()
    return len(left_parentheses) == 0


s = "()"
print(checkValidString(s))  # Output: True






                  #Answer -------- 3


def minDistance(word1, word2):
    n, m = len(word1), len(word2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Fill in the first row and column
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    # Fill in the rest of the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1)

    return dp[n][m]

word1 = "sea"
word2 = "eat"
print(minDistance(word1, word2))  # Output: 2









                  #Answer -------- 4


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def constructTree(s, start, end):
    if start > end:
        return None

    # Find the root value
    rootValue = 0
    i = start
    while i <= end and s[i] != '(':
        rootValue = rootValue * 10 + int(s[i])
        i += 1

    root = TreeNode(rootValue)

    # Find the indices of the left subtree
    leftStart = i + 1
    leftEnd = leftStart
    count = 1
    while count > 0:
        if s[leftEnd] == '(':
            count += 1
        elif s[leftEnd] == ')':
            count -= 1
        leftEnd += 1

    # Construct the left subtree
    root.left = constructTree(s, leftStart, leftEnd - 2)

    # Construct the right subtree if it exists
    if leftEnd <= end:
        rightStart = leftEnd + 2
        rightEnd = end - 1
        root.right = constructTree(s, rightStart, rightEnd)

    return root

def treeToString(root):
    if root is None:
        return ""

    result = str(root.val)
    if root.left or root.right:
        result += '(' + treeToString(root.left) + ')'
        if root.right:
            result += '(' + treeToString(root.right) + ')'

    return result

def constructFromString(s):
    n = len(s)
    if n == 0:
        return None

    return constructTree(s, 0, n - 1)

def preorderTraversal(root):
    result = []
    if root:
        result.append(root.val)
        result.extend(preorderTraversal(root.left))
        result.extend(preorderTraversal(root.right))
    return result





                  #Answer -------- 5


def compress(chars):
    read = 0
    write = 0
    n = len(chars)

    while read < n:
        char = chars[read]
        count = 0

        while read < n and chars[read] == char:
            read += 1
            count += 1

        chars[write] = char
        write += 1

        if count > 1:
            for digit in str(count):
                chars[write] = digit
                write += 1

    return write

chars = ["a","a","b","b","c","c","c"]
print(compress(chars))  # Output: 6
print(chars[:6])  # Output: ["a","2","b","2","c","3"]





                  #Answer -------- 6

from collections import Counter

def findAnagrams(s, p):
    result = []
    len_s, len_p = len(s), len(p)
    freq_p = Counter(p)
    freq_window = Counter(s[:len_p])

    left, right = 0, len_p - 1

    while right < len_s:
        if freq_p == freq_window:
            result.append(left)

        freq_window[s[left]] -= 1
        if freq_window[s[left]] == 0:
            del freq_window[s[left]]
        left += 1

        right += 1
        if right < len_s:
            freq_window[s[right]] += 1

    return result

s = "cbaebabacd"
p = "abc"
print(findAnagrams(s, p))  # Output: [0, 6]




                      #Answer -------- 7



def decodeString(s):
    stack = []
    current_string = ""
    current_number = 0

    for char in s:
        if char.isdigit():
            current_number = current_number * 10 + int(char)
        elif char == '[':
            stack.append(current_string)
            stack.append(current_number)
            current_string = ""
            current_number = 0
        elif char == ']':
            num = stack.pop()
            prev_string = stack.pop()
            current_string = prev_string + num * current_string
        else:
            current_string += char

    return current_string


s = "3[a]2[bc]"
print(decodeString(s))  # Output: "aaabcbc"





                       #Answer -------- 8



def buddyStrings(s, goal):
    if len(s) != len(goal):
        return False

    if s == goal:
        seen = set()
        for char in s:
            if char in seen:
                return True
            seen.add(char)
        return False

    differences = []
    for i in range(len(s)):
        if s[i] != goal[i]:
            differences.append(i)

    return len(differences) == 2 and s[differences[0]] == goal[differences[1]] and s[differences[1]] == goal[differences[0]]



s = "ab"
goal = "ba"
print(buddyStrings(s, goal))  # Output: True





