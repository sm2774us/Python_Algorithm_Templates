# Why did I compile this ?

--------------------------

### **Resources on competitive programming on Python is limited.**

Most of the competitive programmers write in C++ and Java, for their speed in execution. Moreover, examples in Python shown on GeeksForGeeks is transliterated from C++/Java, which ends up verbose and difficult to modify.

### **Python has generally been considered the best introductory language to programming.**

I feel that it is also the best introductory language to competitive programming. Beginner competitive programmers like me are interested in learning the logic. We also want to minimize the time to deal with the syntax and types.

--------------------------

This serves as a resource to bridge Python programmers into competitive programming.

Moreover, if you have been doing competitive programming in other languages, this can serve as a simple guide for you to familiarise yourself with Python.

As with many guides, this is not a complete guide to success from scratch.

* This is not a guide for getting started in Python. It assumes that you already know how to use the basic built-in Data Structures
    * String `st = "abcd"`
    * List `arr = [4,5,6,7,8,5]`
    * Dictionaries `d = {}`
* This is not an introduction to data structures and algorithms. 
  It is best if you have a rough idea of the big-O notation. Otherwise, think of the big-O as the type 
  of increase of space or time required by the computation. This understanding is useful to help you predict 
  whether your algorithm can finish running in a specified time and space. For more information refer to the
  [reddit link](https://www.reddit.com/r/learnprogramming/comments/3gpvyx/algorithms_and_data_structures_cheat_sheets/), 
  and, for a simple cheatsheet refer to [Algorithms-DataStructures-BigONotation](http://cooervo.github.io/Algorithms-DataStructures-BigONotation/index.html). 
* This does not teach you how to handle the full recruitment process, but [Yangshun's guide](https://yangshun.github.io/tech-interview-handbook/) 
  can give you a full overview.

Once you master the special data structures in Python, **you should be able to solve 2.5 out of 4 Leetcode problems every contest** on average, placing you in the top 1000s.

### `list` and its alternatives

--------------------------------

[Documentation](https://docs.python.org/3/tutorial/datastructures.html) The following are the basic operations on a python `list`.

* Reading from a list given an index (aka random access) `arr[4]`
* Appending to the list from right `arr.append(1)`
* Popping from the list from right `arr.pop()`

The above operations take `O(1)` time. You can modify a list in other ways as follows:

* Popping from the list from left `del arr[0]`
* Reversing an array `arr = arr[::-1]`
* Appending to the list from left `arr.insert(0, x)`

The operations take `O(n)` time to run. If you want it faster, you have to use `deque`, which will be explained.

--------------------------------

#### Built-in functions and concepts

--------------------------------

The following are the best practices to make use of lists in concise.

`enumerate`

--------------------------------

[Documentation](https://docs.python.org/3/library/functions.html#enumerate) You want to track both the index and element when iterating through the array.

```python
arr = [3,1,4,1,5,9,2,6,5,3,5,9]
for i in range(len(arr)):
  arr[i] = arr[i] + i

arr # [3, 2, 6, 4, 9, 14, 8, 13, 13, 12, 15, 20]
```

You can use `enumerate` instead

```python
arr = [3,1,4,1,5,9,2,6,5,3,5,9]
for i,ar in enumerate(arr):
  arr[i] = ar + i

arr # [3, 2, 6, 4, 9, 14, 8, 13, 13, 12, 15, 20]
```

The benefit is to avoid the use of awkward `range(len())`. 
Moreover, without the need to use `arr[i]` in the loop you can reduce the nesting 
and make the code clearer.

--------------------------------

#### List comprehensions

--------------------------------
[Documentation](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions) You can iterate through a list with one line of code.

For example, you want to convert a list of integers to a list of strings.

```python
arr = [3,1,4,1,5,9,2]
for i,ar in enumerate(arr):
  arr[i] = str(ar)

arr # ['3', '1', '4', '1', '5', '9', '2']
```

You can do this in one line with a list comprehension.

```python
arr = [3,1,4,1,5,9,2]
arr2 = [str(ar) for ar in arr]
```

You can imbue logic in list comprehensions.

```python
arr3 = [str(ar) for ar in arr if ar > 5]
arr4 = [str(ar) if ar > 5 else "x" for ar in arr]
```

--------------------------------

#### 2D list

--------------------------------

Due to the way Python is structured, beginners usually make the mistake unknowingly.

```python
arr = [0]*5
arr[1] = 9
arr # [0, 9, 0, 0, 0] ok

arr2 = [[0]]*5
arr2[1][0] = 9
arr2 # [[9], [9], [9], [9], [9]]
```

This is how you should set up a 2D list. This behavior has been discussed on [stackoverflow](https://stackoverflow.com/questions/240178/list-of-lists-changes-reflected-across-sublists-unexpectedly).

```python
arr3 = [[0 for _ in range(1)] for _ in range(5)]
arr3[1][0] = 9
arr3 # [[0], [9], [0], [0], [0]]
```

For matrix operations, you may need to use the numpy package ___[ NOTE: not in a coding interview setting though ]___. 
When the matrix is defined to be a number (i.e. a numpy float), numeric operations on the matrix can be faster. 
However, this tutorial will not cover the use of packages from numpy.

#### **Examples:**

* [LeetCode - Problem 1260 - Shift 2D Grid](https://leetcode.com/contest/weekly-contest-163/problems/shift-2d-grid)
  * for those who didn't understand how `sum(grid,[ ])` works.
  * `sum()` takes two parameters `iterable` and `start`, i.e., `sum(iterable,start)`
  * by default `start=0`. So we cannot concatenate a list and integer, 
    that's why we are taking `start=[ ]` empty list
```python
from typing import List

import unittest

class Solution:
    # Solution 1 : Functional Approach
    def shiftGrid_functional_solution_1(self, grid: List[List[int]], k: int) -> List[List[int]]:
        flat = [i for row in grid for i in row]
        k = k % len(flat)
        shift = lambda lst, x: lst[-x:] + lst[:-x]
        reshape = lambda lst, c: [lst[i:i+c] for i in range(0, len(lst), c)]
        return reshape(shift(flat, k), len(grid[0]))

    # Solution 2 : Same idea as above, using sum and slicing 
    #
    #
    # for those who didn't understand how `sum(grid,[ ])` works.
    # sum() takes two parameters iterable and start,i.e.,`sum(iterable,start)`
    # by default start=0. so we cannot concatenate a list and integer,that why we are taking start=[ ] empty list
    #
    def shiftGrid_using_sum_and_slicing_solution_2(self, grid: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(grid), len(grid[0])
        ## flatten the 2-d list
        res = sum(grid, [])
        ## if need to shift multiple times, use mod if k is larger then length of m * n 
        k = k % len(res)
        ## shift function, choose the last k elements and put them in the head
        res = res[-k:] + res[:-k] 
        ## revert back to nested list
        # return list(map(lambda x: res[m*x:(x+1)*m], range (n)))
        return [res[i*n:(i+1) * n] for i in range(m)]

    # Solution 3 : Reverse Approach 
    #
    # Algorithm Steps :
    # 1) put the matrix row by row to a vector.
    # 2) rotate the vector k times.
    # 3) put the vector to the matrix back the same way.
    #
    # The second step is the same as the [LeetCode - Problem 189 - Rotate an Array](https://leetcode.com/problems/rotate-array/),
    # and can be solved in many ways, but here we consider an approach that is simple and
    # has reasonable time and space complexities => the approach is called => "Reverse Approach".
    #
    # An example of the Reverse Approach in action :
    # reverse all elements : [7, 6, 5, 4, 3, 2, 1]
    # reverse first k elements : [5, 6, 7, 4, 3, 2, 1]
    # reverse last len(vec)-k elements : [5, 6, 7, 1, 2, 3, 4]
    # 
    def shiftGrid_using_reverse_approach_solution_3(self, grid: List[List[int]], k: int) -> List[List[int]]:
        # dimensions:
        NR = len(grid)
        NC = len(grid[0])
        vec = [0] * NR * NC #initialize the vector.
        # If k is greater than the length of vector, 
		# the shift will repeat itself in a cycle; 
		# hence, we only care about the remainder.
        k = k % (NR * NC)
		
        #step 1: put the matrix row by row to the vector.
        for i in range(NR):
            for j in range(NC):
                vec[i * NC + j] = grid[i][j]
				
        #step 2: rotate vector k times by reverse approach.
        self.reverse(vec, 0, NR * NC - 1) #reverse all elements.
        self.reverse(vec, 0, k-1)         #reverse first k elements.
        self.reverse(vec, k, NR * NC - 1) #revere last len(vec)-k elements. 
        
        #step 3: put the vector to the matrix back the same way.
        for i in range(NR):
            for j in range(NC):
                grid[i][j] = vec[i * NC + j]
        return grid

    # This function returns the reverse a subset of the vector,
	# bound by "left" and "right" elements
    def reverse(self, vec: List[int], left: int, right: int) -> List[int]:
        while left < right:
            vec[left], vec[right] = vec[right], vec[left]
            left += 1 
            right -= 1


class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_shiftGrid(self) -> None:
        sol = Solution()
        for grid, k, solution in (
            [[[1,2,3],[4,5,6],[7,8,9]], 1, [[9,1,2],[3,4,5],[6,7,8]]],
            [[[3,8,1,9],[19,7,2,5],[4,6,11,10],[12,0,21,13]], 4, [[12,0,21,13],[3,8,1,9],[19,7,2,5],[4,6,11,10]]],
            [[[1,2,3],[4,5,6],[7,8,9]], 9, [[1,2,3],[4,5,6],[7,8,9]]]
        ):
            self.assertEqual(solution, sol.shiftGrid_functional_solution_1(grid, k))
            self.assertEqual(solution, sol.shiftGrid_using_sum_and_slicing_solution_2(grid, k))
            self.assertEqual(solution, sol.shiftGrid_using_reverse_approach_solution_3(grid, k))

if __name__ == "__main__":
    unittest.main()
```

--------------------------------

### `deque` object

--------------------------------

[Documentation](https://docs.python.org/3/library/collections.html#collections.deque) You might want to append 
to both sides of a list. `deque` is a doubly-linked list.

> `deque` is a list-like container with fast appends and pops on either end.
>

```python
from collections import deque
de = deque(arr)
```

Popping and appending to the list on either side is now `O(1)`.

```python
de.append(1)
de.pop() # 1
de.appendleft(0)
de.popleft() # 0
```

Reading from the list (aka random access) `de[3]` now takes `O(n)` time.

#### **Examples:**

* [LeetCode - Problem 862 - Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/)
```python
from typing import List
import collections

import unittest

class Solution:
    def shortestSubarray(self, A: List[int], K: int) -> int:
        d = collections.deque([[0, 0]])
        res, cur = float('inf'), 0
        for i, a in enumerate(A):
            cur += a
            while d and cur - d[0][1] >= K:
                res = min(res, i + 1 - d.popleft()[0])
            while d and cur <= d[-1][1]:
                d.pop()
            d.append([i + 1, cur])
        return res if res < float('inf') else -1
    

class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_shortestSubarray(self) -> None:
        sol = Solution()
        for A, K, solution in (
            [[1], 1, 1],
            [[1,2], 4, -1],
            [[2,-1,2], 3, 3]
        ):
            self.assertEqual(solution, sol.shortestSubarray(A, K))

if __name__ == "__main__":
    unittest.main()
```
* [LeetCode - Problem 1298 - Maximum Candies You Can Get from Boxes](https://leetcode.com/contest/weekly-contest-168/problems/maximum-candies-you-can-get-from-boxes/)
  * Solution (1) -> using `deque`
  * Solution (2) -> using `list`->`extend`
```python
from typing import List
from collections import deque

import unittest

class Solution:
    # Solution 1: Using deque
    def maxCandies_using_deque_solution_1(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:        
        q = deque(initialBoxes)
        visited = set()
        res = 0
        while q:
            itr, opened = len(q), False  # To detect cycle
            opened = False
            while (itr):
                itr -= 1
                v = q.popleft()
                if status[v]: # Open box, (key is available or is open)
                    
                    opened = True
                    res += candies[v]
                    visited.add(v)
                    
                    for x in keys[v]:
                        status[x] = 1
                    
                    for x in containedBoxes[v]:
                        if x not in visited:
                            q.append(x)
                            
                elif v not in visited: # Open when key is available
                    q.append(v)
            if not opened:
                return res  # Exit cycle detected
        return res

    # Solution 2: Using `list`->`extend`
    def maxCandies_using_list_extend_solution_2(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:
        boxes = initialBoxes
        res = 0
        while True:
            opened = False
            for i, box in enumerate(boxes):
                if box != -1 and status[box]: # If the box is unvisited and open
                    res += candies[box]
                    for key in keys[box]: # You have the key, open now or when the box is available
                        status[key] = 1
                    boxes.extend(containedBoxes[box]) # You have found new boxes to be scanned next
                    boxes[i] = -1 # Make this box visited
                    opened = True
            if not opened: # Exit if no boxes can be opened
                break
        return res
    

class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_maxCandies(self) -> None:
        sol = Solution()
        for status, candies, keys, containedBoxes, initialBoxes, solution in (
            [[1,0,1,0], [7,5,4,100], [[],[],[1],[]], [[1,2],[3],[],[]], [0], 16],
            [[1,0,0,0,0,0], [1,1,1,1,1,1], [[1,2,3,4,5],[],[],[],[],[]], [[1,2,3,4,5],[],[],[],[],[]], [0], 6],
            [[1,1,1], [100,1,100], [[],[0,2],[]], [[],[],[]], [1], 1],
            [[1], [100], [[]], [[]], [], 0],
            [[1,1,1], [2,3,2], [[],[],[]], [[],[],[]], [2,1,0], 7]
        ):
            self.assertEqual(solution, sol.maxCandies_using_deque_solution_1(status, candies, keys, containedBoxes, initialBoxes))
            self.assertEqual(solution, sol.maxCandies_using_list_extend_solution_2(status, candies, keys, containedBoxes, initialBoxes))

if __name__ == "__main__":
    unittest.main()
```

--------------------------------

### `bisect` **functions**

--------------------------------

[Documentation](https://docs.python.org/3/library/bisect.html) You have a list you know is sorted. 
You are given a new number, and you want to find the first index which element is greater 
than or equal to the new number.

```python
arr = [1,1,2,3,4,5,5,5,5,9]
x = 5
```

One way is to iterate through the entire array.

```python
for index,ar in enumerate(arr):
  if ar >= x:
    print(index)
    break
```

Instead of "greater than or equal to" you want the first element that is "greater than".

```python
for index2,ar in enumerate(arr):
  if ar > x:
    print(index2)
    break
else:  # this is a for-else loop
  print(index2)
```

The other way is to implement a binary search, and I will leave this as an exercise for the reader.

Instead of implementing the binary search yourself, you can use the functions from `bisect`. 
However, please be prepared to implement a binary search from scratch, as it may be required for interview and custom problems.

```python
import bisect
index = bisect.bisect_left(arr, x)
index # 5
```

This is for the case when you want the first element "greater than" rather than "greater than or equal to".

```python
index2 = bisect.bisect_right(arr, x)
index2 # 9
```

#### **Examples:**

Certain database questions simply require one to know the existence of `bisect`.

___[ NOTE: In a real coding interview don't use the python built-in `bisect` library, since it doesn't demonstrate
any knowledge of binary search itself. ]___

In both the examples below 2 solutions, per problem, are presented -> (1) using `bisect`, 
and, (2) without using `bisect` and implementing binary search [ choose (2) for your coding interview ].

* [LeetCode - Problem 34 - Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
```python
from typing import List
from typing import Tuple

import bisect
import operator

import unittest

class Solution:
    # Solution 1: Binary search using the python built-in library `bisect`
    def searchRange_using_bisect_solution_1(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1, -1]
        
        left,right = bisect.bisect_left(nums, target), bisect.bisect_right(nums, target)
                
        return [left, right - 1] if left < right else [-1, -1]

    # Solution 2: Binary search without using the python built-in library `bisect`
    def searchRange_binary_search_solution_2(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1, -1]
        if len(nums) == 1 and nums[0] == target: return [0, 0]
        
        _, high = self.binary_search_half_an_array(nums, target)        
        low, _ = self.binary_search_half_an_array(nums, target, direction="left")
        
        if low > high: return [-1, -1]  # can't find it   
        return [low, high]

    def binary_search_half_an_array(self, nums: List[int], target: int, direction: str ='right') -> Tuple(int, int):
        low = 0
        high = len(nums) - 1
        
        if direction == "right":
            op = operator.le    # <=
        else:
            op = operator.lt    # < 
            
        while low <= high:
            mid = (low + high) / 2
            val = nums[mid]
            if op(val, target): low = mid + 1
            else: high = mid - 1
            
        return low, high
    

class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_searchRange(self) -> None:
        sol = Solution()
        for nums, target, solution in (
            [[5,7,7,8,8,10], 8, [3,4]],
            [[5,7,7,8,8,10], 6, [-1,-1]],
            [[], 0, [-1,-1]]
        ):
            self.assertEqual(solution, sol.searchRange_using_bisect_solution_1(nums, target))
            self.assertEqual(solution, sol.searchRange_binary_search_solution_2(nums, target))

if __name__ == "__main__":
    unittest.main()
```
* [LeetCode - Problem 475 - Heaters](https://leetcode.com/problems/heaters/)
```python
from typing import List
from typing import Tuple

import bisect
import operator

import unittest

class Solution:
    # Solution 1: Binary search using the python built-in library `bisect`
    def findRadius_using_bisect_solution_1(self, houses: List[int], heaters: List[int]) -> int:
        heaters.sort()
        return max(min(abs(house - heater)
                       for i in [bisect.bisect(heaters, house)]
                       for heater in heaters[i-(i>0):i+1])
                   for house in houses)
        
    # Solution 2: Binary search without using the python built-in library `bisect`
    #
    # I agree that the bisect is useful and powerful, 
    # but I would like to try to implement it with plain python 
    # to see how complicated it is.
    #
    # 1. First, we sort the house position array and the heater position array.
    #
    # 2. Second, for each house, we make 2 check if it does not locate between two adjecent heaters. If they does not, we could calculate the needed radius with max function.
    #
    # 3. Third, if the house between 2 heaters, we use binary search to locate the position.
    #    With the left and right pointer, we could get the mid pointer.
    #
    #       3.1) If the house locates just on some heater's position, we do nothing.
    #
    #       3.2) If the house position is lesser than the heaters[mid], we do a further check if the mid + 1 falls in valid range, and check if heater[mid + 1] > h to know if the house is 
    #            surrounded by heater[mid] and heater[mid + 1]. If not, just let the binary search continue.
    #            And the same idea for if the house position is greater than the heaters[mid].
    #
    #       + Note that the house must locate between two heaters, so the checking process in the binary search is good enough to get the correct minimum radius for every house.
    # * We keep the left pointer among all the house checking processes, because the left pointer do increase for our house positions is in an ascending order. 
    #   This trick did provide some speeding up.
    #
    def findRadius_binary_search_solution_2(self, houses: List[int], heaters: List[int]) -> int:
        heaters.sort()
        houses.sort()
        rad = 0
        l = 0
        for h in houses:
            if h < heaters[0]:
                rad = max(rad, heaters[0] - h)
            elif h > heaters[-1]:
                rad = max(rad, h - heaters[-1])
            else:
                r = len(heaters) - 1
                while l <= r:
                    mid = (l + r) // 2
                    if heaters[mid] == h:
                        break
                    if h < heaters[mid]:
                        if mid - 1 >= 0 and heaters[mid - 1] < h:
                            rad = max(rad, min(h - heaters[mid-1], heaters[mid] - h))
                            break
                        r = mid - 1
                    else:
                        if mid + 1 < len(heaters) and heaters[mid + 1] > h:
                            rad = max(rad, min(h - heaters[mid], heaters[mid+1] - h))
                            break
                        l = mid + 1
        return rad
        
class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_findRadius(self) -> None:
        sol = Solution()
        for houses, heaters, solution in (
            [[1,2,3], [2], 1],
            [[1,2,3,4], [1,4], 1],
            [[1,5], [2], 3]
        ):
            self.assertEqual(solution, sol.findRadius_using_bisect_solution_1(houses, heaters))
            self.assertEqual(solution, sol.findRadius_binary_search_solution_2(houses, heaters))

if __name__ == "__main__":
    unittest.main()
```
--------------------------------

### `heapq` object

--------------------------------

[Documentation](https://docs.python.org/3/library/heapq.html) You want to track the smallest object in the list, and remove it when needed.

One way is to continually sort the list and remove the smallest element. Sorting takes `O(nlogn)`.

```python
arr = [4,6,7,1]
new = [[1], [], [9,9], [5]]
for ne in new:
  arr.extend(ne)
  arr.sort()
  print(arr)
  del arr[0]
```

The other way is to identify the smallest element and remove it. This takes `O(n)` time for every operation.

However, there is this special data structure that allows the small value. You can find 
several interactive [visualisation](https://visualgo.net/en/heap) on the Internet.

```python
arr = [4,6,7,1]
import heapq
heapq.heapify(arr)
print(arr) # [1, 4, 7, 6]
```

You see that the array is somewhat sorted, but not really. From the documentation:

> Heaps are binary trees for which every parent node has a value less 
> than or equal to any of its children. This implementation uses arrays 
> for which `heap[k] <= heap[2*k+1]` and `heap[k] <= heap[2*k+2]` 
> for all `k`, counting elements from zero. 
> For the sake of comparison, non-existing elements are considered to 
> be infinite. The interesting property of a heap is that 
> its smallest element is always the root, `heap[0]`.
>

```python
heapq.heappush(arr,4) # push an element
popped = heapq.heappop(arr) # returns the smallest element
popped # 1
```

This is a minimum heap. For max-heap, the recommended solution is to multiply and input.

As `heapq` is used to maintain fast popping of the smallest element, 
the baseline method of iterating through the whole array for the smallest element 
will take `O(n)` time.

In API-based questions, you may be tasked to get the smallest element repeatedly 
and doing it in `O(n)` may be too slow. With `heapq`, the pushing and popping of elements 
take `O(logn)` time. Note that creating a heap takes `O(nlogn)` time.

#### **Examples:**

* [LeetCode - Problem 703 - Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)
  * Solution 1: Fixed Size Heap [ O(T): `O(nlgn)` | O(S): `O(n)` ].
  * Solution 2: Alternative: heappushpop.
  * Solution 3: Alternative: heapreplace.
  * Solution 4: Bisect Sort [ O(T): `O(nlgn)` | O(S): `O(n)` ].
```python
from typing import List

import heapq

import unittest

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        heapq.heapify(nums)
        while len(nums) > k: heapq.heappop(nums)
        self.k, self.lst = k, nums

    def add(self, val: int) -> int:
        heapq.heappush(self.lst, val)
        if len(self.lst) > self.k: heapq.heappop(self.lst)
        return self.lst[0]

class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_kthLargest(self) -> None:
        kthLargest = KthLargest(3, [4, 5, 8, 2])
        res = kthLargest.add(3)   # return 4
        self.assertEqual(4, res)
        res = kthLargest.add(5)   # return 5
        self.assertEqual(5, res)
        res = kthLargest.add(10)  # return 5
        self.assertEqual(5, res)
        res = kthLargest.add(9)   # return 8
        self.assertEqual(8, res)
        res = kthLargest.add(4)   # return 8
        self.assertEqual(8, res)

if __name__ == "__main__":
    unittest.main()
```
```python
from typing import List

import heapq

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        heapq.heapify(nums)
        while len(nums) > k: heapq.heappop(nums)
        self.k, self.lst = k, nums

    def add(self, val: int) -> int:
        if len(self.lst) < self.k: heapq.heappush(self.lst, val)
        else: heapq.heappushpop(self.lst, val)
        return self.lst[0]
```
```python
from typing import List

import heapq

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        heapq.heapify(nums)
        while len(nums) > k: heapq.heappop(nums)
        self.k, self.lst = k, nums

    def add(self, val: int) -> int:
        if len(self.lst) < self.k: heapq.heappush(self.lst, val)
        elif val > self.lst[0]: heapq.heapreplace(self.lst, val)
        return self.lst[0]
```
```python
from typing import List

import bisect

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k, self.lst = k, sorted(nums)[-k:]

    def add(self, val: int) -> int:
        bisect.insort(self.lst, val)
        if len(self.lst) > self.k: del self.lst[0]
        return self.lst[0]

```
* [LeetCode - Problem 23 - Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
  * In a coding interview go for - Solution 2 - Divide and Conquer approach. 
```python
from typing import List

import heapq

import unittest

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def initList(self, nums):
        if not nums:
            return None
        head = None
        current = None

        for n in nums:
            if not head:
                head = ListNode(n)
                current = head
            else:
                node = ListNode(n)
                current.next = node
                current = node
        return head

class Solution:
    def mergeKLists_using_heapq_solution_1(self, lists: List[ListNode]) -> ListNode:
        # Monkey-patch the ListNode to support < operator
        ListNode.__lt__ = lambda self, other: self.val < other.val
        #Not needed
        #ListNode.__eq__ = lambda self, other: self.val == other.val
        # Filter empty lists and handle case where all lists are empty
        heap = [l for l in lists if l]
        if not heap: return None
        # Turn "heap" into a real heap
        heapq.heapify(heap)
    
        def pop_push():
            ''' Pop the min node from heap and push its next node, if any '''
            node = heapq.heappop(heap)
            if node.next: heapq.heappush(heap, node.next)
            return node
    
        # Pop all nodes from the heap into a new list 
        head = tail = pop_push()
        while heap: # O(N)
            tail.next = pop_push() # O(log K)
            tail = tail.next

        return head
    
    def mergeKLists_using_divide_and_conquer_solution_2(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return None
        if len(lists) == 1:
            return lists[0]
        mid = len(lists) // 2
        l, r = self.mergeKLists_using_divide_and_conquer_solution_2(lists[:mid]), self.mergeKLists_using_divide_and_conquer_solution_2(lists[mid:])
        return self.merge(l, r)

    def merge(self, l: ListNode, r: ListNode) -> ListNode:
        dummy = p = ListNode()
        while l and r:
            if l.val < r.val:
                p.next = l
                l = l.next
            else:
                p.next = r
                r = r.next
            p = p.next
        p.next = l or r
        return dummy.next
        
class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_mergeKListsUsingHeap(self) -> None:
        listNode = ListNode()
        s = Solution()
        self.assertEqual(
            listNode.initList([1, 1, 2, 3, 4, 4, 5, 6]),
            s.mergeKLists_using_heapq_solution_1(
                [
                    listNode.initList([1, 4, 5]),
                    listNode.initList([1, 3, 4]),
                    listNode.initList([2, 6]),
                ]
            ),
        )
        self.assertEqual(listNode.initList([]), s.mergeKLists_using_heapq_solution_1([]))
        self.assertEqual(
            listNode.initList([]), s.mergeKLists_using_heapq_solution_1(listNode.initList([]))
        )
        
    def test_mergeKListsUsingDivideAndConquer(self) -> None:
        listNode = ListNode()
        s = Solution()
        self.assertEqual(
            listNode.initList([1, 1, 2, 3, 4, 4, 5, 6]),
            s.mergeKLists_using_divide_and_conquer_solution_2(
                [
                    listNode.initList([1, 4, 5]),
                    listNode.initList([1, 3, 4]),
                    listNode.initList([2, 6]),
                ]
            ),
        )
        self.assertEqual(listNode.initList([]), s.mergeKLists_using_divide_and_conquer_solution_2([]))
        self.assertEqual(
            listNode.initList([]),
            s.mergeKLists_using_divide_and_conquer_solution_2(listNode.initList([])),
        )

if __name__ == "__main__":
    unittest.main()
```  
* [LeetCode - Problem 826 - Most Profit Assigning Work](https://leetcode.com/problems/most-profit-assigning-work/)
* [LeetCode - Problem 973 - K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)

___Note: Please use quickselect ( solution 4 ) if the interviewer prohibits usage of heapq library___

1. Sort - time O(nlogn), space O(n)
1. Minheap - time O(n + klogn), space O(n)
1. Maxheap - time O(nlogk), space O(k)
1. Quickselect - time O(n) average/O(n^2) worst case, space O(n)

A quick comment on the 2 heap solutions. Most heap solutions on here have discussed the maxheap (approach 3). 
I believe in an interview setting, it is worth bringing up the **minheap** implementation too, 
and it may actually work faster than the **maxheap**, depending on the values of `n` and `K`. 
As this problem states `K <= n`, putting the larger term `n` in the logarithm makes mathematical sense. 
If you plotted both time complexities, you would also observe a crossover in which the **minheap** implementation 
is actually more time efficient than the **maxheap**, given a large enough value of `K` relative to `n`.

**1: Basic sort - time `O(nlogn)`, space `O(n)`**

Basic sort performed on the squared distance from origin. Simply return the first `K` elements after sorting.
```python
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        return sorted(points, key=lambda x: x[0]**2 + x[1]**2)[:K]
```
**2: Minheap - time `O(n + klogn)`, space `O(n)`**

A minheap is created with all `n` elements, and we heappop `K` times to generate the result. 
Heapifying takes `O(n)` and heappopping takes `O(klogn)`, since it is `O(logn)` per pop. 
Space is `O(n)` since the heaps contains all `n` elements.
```python
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        arr = [(p[0] ** 2 + p[1] ** 2, p) for p in points]
        heapq.heapify(arr)
        res = []
        for _ in range(K):
            res.append(heapq.heappop(arr)[1])
        return res
```
**3: Maxheap - time `O(nlog(k))`, space `O(k)`**

A maxheap of size `K` is maintained at all times. For each element in points, the element 
is added to the heap and then the max item is removed (`heapq.heappushpop`). We return the entire 
heap of `K` elements. Since Python's `heapq` implementation of a heap is a `minheap`, 
we multiply the squared distance from origin by -1 before pushing an element to the heap in order 
to mimic the functionality of a `maxheap`. Since a `maxheap` by definition can only pop 
its maximum value and we are maintaining a heap of size `K`, our heap will always hold 
the `K` closest points to the origin while we iterate through `points`.

The precise time compelxity is `O(k + (n-k)log(k))` since it is `O(k)` for heapifying 
the first `K` elements, and for the remaining `n-K` elements, it takes `O(k)` time to process 
each element (adding the element and then popping the largest element).

If we assume `K << n`, the time complexity simplifies to `O(nlogk)`. Space is `O(k)` because 
we are maintaining a heap of size `k`.

```python
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        arr = [(-1 * (p[0] ** 2 + p[1] ** 2), p) for p in points]
        heap = arr[:K]
        heapq.heapify(heap)
        for p in arr[K:]:
            heapq.heappushpop(heap, p)
        return [coord for distance, coord in heap]
```

**4: Quickselect (unstable, K elements are not returned in sorted order)**

Time: average/best O(n), worst O(n^2)

Space: O(n) - sorting is inplace

This approach gave me some difficulty when trying to fully understand it. Having a good understanding of quicksort helps tremendously. A modification on basic quicksort yields quickselect, which can be used if we are only interested in returning the kth smallest element in a list. 
[Here is a good overview of quickselect](https://en.wikipedia.org/wiki/Quickselect).

We base approach 4 on quickselect. Instead of returning only the `K`th smallest element, 
we want to return all elements from index `0:K`.

The `partition` function is the exact same partition function used in regular quicksort. 
I've implemented the `Lomuto` partition, which is simpler to understand than the `Hoare` partition 
in my opinion. Given `arr, l, r`, we take the rightmost element as the pivot, reposition the pivot element 
into its appropriate place within the array, and return the pivot's new index. If any of this is confusing, 
I strongly suggest reading up on the basics of quicksort.

The `quickselect` function, like a general quicksort function would, first checks to make sure 
we are not about to partition an array of size 0 or 1. We swap a random index in the array with the rightmost element, 
since the rightmost element will be used as pivot. This randomized index swap allows the algorithm to perform in O(n) time on average in the case of a sorted array.

Once the `partition` function has returned the partition index, `pIndex`, we compare `pIndex` with `K`. 
Specifically, we are comparing `(K - 1)` with `(pIndex - l)`. We are 0-indexing `K` before we 
make our comparison, and we are comparing with `(pIndex - l)` since this is the relative position of 
`pIndex` in our recursive call.

If `(K - 1) > (pIndex - l)`, this means we need to keep expanding our quickselect toward `pIndex`'s right. 
We make this recursive call, but instead of passing in `K`, we pass in a value of `K` that is offset by the 
number of positions pIndex is relative to `l`.

> Imagine we have an array of distances, `[2, 8, 5, 3]`, and we want to solve for `K=4`. After the first partition 
> call `partition(arr, l=0, r=3)` and assuming we take 3 as the pivot, the array may become something 
> like `[2, 3, 8, 5]`, in which 3 is in its sorted position. `pIndex` will return as `1`. The first if statement 
> will execute, and `quickselect(arr, l=2, r=3, K=2)` will be recursively called. Why `K=2`? 
> Because from indices 2 through 3 in the array, we want its 2 smallest numbers now. 
> We only want its 2 smallest numbers, instead of the original `K=4`, because when `pIndex` returned as value `1`, 
> we already know indices `0` and `1` of the array are the first 2 smallest numbers...now we only need to 
> find 2 more.

If `(K - 1) < (pIndex - l)`, this means we need to finish off quickselecting to `pIndex`'s left. 
We don't need to look to `pIndex`'s right anymore.

> Imagine we have a different array of distances `[2, 8, 3, 5]` and we want to solve for `K=1`. 
> After the first partition call `partition(arr, l=0, r=3)` and assuming we take 5 as the pivot, 
> the array may become something like `[3, 2, 5, 8]`, in which 5 is in its sorted position. `pIndex` will return as 2. 
> The good news is we don't need to look to the right of element 5 anymore at all to solve the problem. 
> What we need to do now is recursively sort the elements to the left of 5.

Finally, we return the first K elements.

To evaluate time complexity, we first understand that like quicksort, our best case is 
when the pivot we select is the median of `array[l:r+1]` and `pIndex` returns an index close 
to the midpoint index between `l` and `r`. If this best case holds true, then at each iteration, 
we are splitting our array in half and picking only one of the halves to recurse upon. 
So the time complexity will be `O(n + 1/2n + 1/4n + ... 1) = O(2n) = O(n)`. This is simply 
a geometric sequence. Because we are randomizing the pivot we use, we can say on average, 
we will be getting this best case of `O(n)`.

```python
from random import randint
class Solution:
    def kClosest(self, points, K):
        arr = [(p[0] ** 2 + p[1] ** 2, p) for p in points]
        self.quickselect(arr, 0, len(arr) - 1, K)
        return [coord for distance, coord in arr[:K]]

    def partition(self, arr, l, r): # Lomuto partition
        pivot = arr[r][0] # distance of right-most element
        i = l
        for j in range(l, r + 1):
            if arr[j][0] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[r], arr[i] = arr[i], arr[r]
        return i

    def quickselect(self, arr, l, r, K):
        if l < r:
            rand_idx = randint(l, r) # random pivot index
            arr[rand_idx], arr[r] = arr[r], arr[rand_idx] # swap random pivot index with rightmost element
            pIndex = self.partition(arr, l, r)
            if (K - 1) > (pIndex - l): # we need to work on elements to pIndex's right 
                self.quickselect(arr, pIndex + 1, r, (K - 1) - (pIndex - l))
            elif (K - 1) < (pIndex - l): # we need to sort the elements to pIndex's left
                self.quickselect(arr, l, pIndex - 1, K)

```

--------------------------------

### `dict` and its subclasses

--------------------------------

[Documentation](https://docs.python.org/3/tutorial/datastructures.html) A dictionary is a hashmap of key-value pairs.

* Creating a dictionary `d={}`
* Populating a key with a value `d["foo"] = "bar"`
* To check if a dictionary has a key `foo in d`
* Deleting a key as well as its value `del d["foo"]`

The above operations take `O(1)` time.

You can obtain the keys by converting the dictionary into a list.

```python
d = {"foo":"bar", "bar":"baz"}
list(d)
sorted(d)
```

If you want the values, you have to iterate through its keys.

```python
[(k,v) for k,v in d.items()]
```

I hope you understand how to use the dictionary.

Some procedures using a dictionary can be implemented with less code. 
Concise code is more understandable and less prone to mistakes.

--------------------------------

#### `defaultdict`

[Documentation](https://docs.python.org/3/library/collections.html#collections.defaultdict) For example, you have a 
list of directed edges you want to populate this into a dictionary.

```python
edges = [[1,2], [2,1], [1,3]]
d = {}
for start, end in edges:
  if start in d:
    d[start].append(end)
  else: 
    d[start] = [end]

d # {1: [2, 3], 2: [1]}
```

You may use `defaultdict` to skip initializing the value for every key.

```python
from collections import defaultdict
d = defaultdict(list)
for start, end in edges:
  d[start].append(end)
  
d # defaultdict(<class 'list'>, {1: [2, 3], 2: [1]})
```

This makes your code neater. However, it assumes that every value is a list. This is useful in helping 
to construct graphs from a list of edges. Graphs are usually necessary to solve problems involving paths. 
This is an example.

#### **Examples:**

[LeetCode - Problem 1219 - Path with Maximum Gold](https://leetcode.com/problems/path-with-maximum-gold/)

Because the grid is so small, and the number of cells that could contain gold is even smaller, 
backtracking is a suitable strategy to use.

This code uses the strategy and template you can find in the Explore section on backtracking.
https://leetcode.com/explore/learn/card/recursion-ii/472/backtracking/2654/

```python
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:

		# Given a row and a column, what are all the neighbours?
        def options(row, col):
            if row > 0:
                yield(row - 1, col)
            if col > 0:
                yield(row, col - 1)
            if row < len(grid) - 1:
                yield(row + 1, col)
            if col < len(grid[0]) - 1:
                yield(row, col + 1)
        
		# Keep track of current gold we have, and best we've seen.
        self.current_gold = 0
        self.maximum_gold = 0
        
        def backtrack(row, col):
		
			# If there is no gold in this cell, we're not allowed to continue.
            if grid[row][col] == 0:
                return
			
			# Keep track of this so we can put it back when we backtrack.
            gold_at_square = grid[row][col] 
            
			# Add the gold to the current amount we're holding.
			self.current_gold += gold_at_square
			
			# Check if we currently have the max we've seen.
            self.maximum_gold = max(self.maximum_gold, self.current_gold)
			
			# Take the gold out of the current square.
            grid[row][col] = 0
			
			# Consider all possible ways we could go from here.
            for neigh_row, neigh_col in options(row, col):
				# Recursively call backtrack to explore this way.
                backtrack(neigh_row, neigh_col)
			
			# Once we're done on this path, backtrack by putting gold back.
            self.current_gold -= gold_at_square
            grid[row][col] = gold_at_square 
		
		# Start the search from each valid starting location.
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                backtrack(row, col)
		
		# Return the maximum we saw.
        return self.maximum_gold
```

--------------------------------

#### `Counter`

[Documentation](https://docs.python.org/3/library/collections.html#collections.Counter) For example, you want to count 
the number of each element in a list.

```python
from collections import defaultdict

digits = [3,1,4,1,5,9,2,6,5,3,5,9]
d = defaultdict(int)
for digit in digits:
  d[digit] += 1
  
d # defaultdict(<class 'int'>, {3: 2, 1: 2, 4: 1, 5: 3, 9: 2, 2: 1, 6: 1})
```

There is a function `Counter` which does the work in one line after importing.

```python
from collections import Counter
d = Counter(digits)

d # Counter({5: 3, 3: 2, 1: 2, 9: 2, 4: 1, 2: 1, 6: 1})
```

#### **Examples:**

* [LeetCode - Problem 1297 - Maximum Number of Occurrences of a Substring](https://leetcode.com/contest/weekly-contest-168/problems/maximum-number-of-occurrences-of-a-substring/)
```python
from typing import List
import collections

import unittest

class Solution:
    def maxFreq(self, s: List[int], maxLetters: int, k: int, maxSize: int) -> int:
        count = collections.Counter(s[i:i + k] for i in range(len(s) - k + 1))
        return max([count[w] for w in count if len(set(w)) <= maxLetters] + [0])

class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_maxFreq(self) -> None:
        sol = Solution()
        for s, maxLetters, minSize, maxSize, solution in (
            ["aababcaab", 2, 3, 4, 2],
            ["aaaa", 1, 3, 3, 2],
            ["aabcabcab", 2, 2, 3, 3],
            ["abcde", 2, 3, 3, 0] 
        ):
            self.assertEqual(solution, sol.maxFreq(s, maxLetters, minSize, maxSize))

if __name__ == "__main__":
    unittest.main()
```
* [LeetCode - Problem 1296 - Divide Array in Sets of K Consecutive Numbers](https://leetcode.com/contest/weekly-contest-168/problems/divide-array-in-sets-of-k-consecutive-numbers/)
```python
from typing import List
import collections

import unittest

class Solution:
    def isPossibleDivide(self, nums: List[int], k: int) -> bool:
        count = collections.Counter(nums)
        for start in sorted(nums):
            if count[start] != 0:  # values with occurrences == 0 have already been used in a previous group of k consecutive nums
                occ = count[start]
                for v in range(start, start + k):
                    count[v] -= occ
                    if count[v] < 0:  # a number that should have existed to make k consecutive numbers is not there
                        return False
        return True

class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_isPossibleDivide(self) -> None:
        sol = Solution()
        for nums, k, solution in (
            [[1,2,3,3,4,4,5,6], 4, True],               # Array can be divided into [1,2,3,4] and [3,4,5,6].
            [[3,2,1,2,3,4,3,4,5,9,10,11], 3, True],     # Array can be divided into [1,2,3] , [2,3,4] , [3,4,5] and [9,10,11].
            [[3,3,2,2,1,1], 3, True],
            [[1,2,3,4], 3, False]                       # Each array should be divided in subarrays of size 3. 
        ):
            self.assertEqual(solution, sol.isPossibleDivide(nums, k))

if __name__ == "__main__":
    unittest.main()
```

[LeetCode - Problem 1296 - Divide Array in Sets of K Consecutive Numbers - Solution Explanation Video](https://www.youtube.com/watch?v=r4VZCwHs4Zo)

![Image](https://img.youtube.com/vi/r4VZCwHs4Zo/default.jpg)

--------------------------------

### `set`

--------------------------------

[Documentation](https://docs.python.org/3/library/stdtypes.html#set) Sometimes you want a dictionary without its values. 
We present the python `set`.

You can create a set from a list, or define it in non-empty curly braces.

```python
set1 = set(["one-only", "one-two"])
set2 = {"two-only", "one-two"}
```

Like a list, you can iterate through a set and return its length

```python
print(len(set1))
for x in set1: print(x)
```

You can check whether an element is in a set

```python
"one-two" in set1
```

The use of a set can help you filter the unique items of a list

```python
arr = [1,1,2,3,3]
arr = list(set(arr))
```

You can add elements to a set (or multiple elements with update)

```python
set1.add("orange")
set1.update(["apple", "orange"])
```

You can delete an item from a set.

```python
set1.remove("apple")
```

You can find the union of two sets. In the following, the elements in set3 is made up of elements that appear either in set1 or set2 or both.

```python
set3 = set1.union(set2)
set3 = set1 | set2
set3 # {'one-only', 'two-only', 'one-two', 'orange'}
```

You can find the intersection of two sets. In the following, the elements in set3 is made up of elements that appear either in both set1 and set2.

```python
set3 = set1.intersection(set2)
set3 = set1 & set2
set3 # {'one-two'}
```

You can take the difference of one set from another.

```python
set3 = set1.difference(set2)
set3 = set1 - set2
# {'one-only', 'orange'}
```

You can take the union and exclude the intersection of a pair of sets.

```python
set3 = set1.symmetric_difference(set2)
set3 = set1 ^ set2
# {'two-only', 'one-only', 'orange'}
```
--------------------------------

#### **Examples:**

[LeetCode - Problem 1239 - Maximum Length of a Concatenated String with Unique Characters](https://leetcode.com/contest/weekly-contest-160/problems/maximum-length-of-a-concatenated-string-with-unique-characters/)

**Explanation**

1. Initialize the result `res` to include the case of empty string "".

   `res` includes all possible combinations we find during we iterate the input.
1. Iterate over the input strings, but skip the word that have duplicate characters.
   The examples are kind of misleading, but the input string can have duplicate characters,
   no need to consider these strings.
1. For each string,
   we check if it conflicts with the combination that we found.
   
   If they have intersection of characters, we skip it.
   
   If not, we append this new combination to result.
1. Return the maximum length from all combinations.

```python
from typing import List

import unittest

class Solution:
    def maxLength(self, arr: List[str]) -> int:
        dp = [set()]
        for a in arr:
            if len(set(a)) < len(a): continue
            a = set(a)
            for c in dp[:]:
                if a & c: continue
                dp.append(a | c)
        return max(len(a) for a in dp)
```

--------------------------------

### Contest examples

--------------------------------

Explanation of column headers
* Points - number of points on the problem
* Concept - what concepts covered. (Basic) suggests that understanding the problem and the basic data structures is sufficient to solve the problem.


| **Contest / Problem**	| **Points** | **Concepts** |
| ---- | ---- | ---- |
| **Weekly Contest 168** | | |
| [LeetCode - Problem 1296 - Divide Array in Sets of K Consecutive Numbers](https://leetcode.com/contest/weekly-contest-168/problems/divide-array-in-sets-of-k-consecutive-numbers/) | 4 | `Counter` |
| [LeetCode - Problem 1297 - Maximum Number of Occurrences of a Substring](https://leetcode.com/contest/weekly-contest-168/problems/maximum-number-of-occurrences-of-a-substring/)	| 6	| `Counter`, `set` |
| [LeetCode - Problem 1298 - Maximum Candies You Can Get from Boxes](https://leetcode.com/contest/weekly-contest-168/problems/maximum-candies-you-can-get-from-boxes/) | 7 | `deque`, `list` |
| **Weekly Contest 163** | | |
| [LeetCode - Problem 1260 - Shift 2D Grid](https://leetcode.com/contest/weekly-contest-163/problems/shift-2d-grid) | 3 | `2D list` |
| **Weekly Contest 161** | | |		
| [LeetCode - Problem 1247 - Minimum Swaps to Make Strings Equal](https://leetcode.com/contest/weekly-contest-161/problems/minimum-swaps-to-make-strings-equal) | 4 | `defaultdict` |
| **Weekly Contest 160** | | |		
| [LeetCode - Problem 1239 - Maximum Length of a Concatenated String with Unique Characters](https://leetcode.com/contest/weekly-contest-160/problems/maximum-length-of-a-concatenated-string-with-unique-characters/) | 5 | `set` |

--------------------------------

### References

--------------------------------

I have referred to these [coding notes](python_coding_notes/05_Python3-DataStructures.md). 
For this writeup I added examples on how the same (but less neat or efficient) could be done 
with basic data structures in Python, and the relevant leetcode contest questions that this has helped.

As mentioned in the introduction, [Yangshun](https://yangshun.github.io/tech-interview-handbook/) 
provides a comprehensive guide for the full technical interview process.

--------------------------------

### Conclusion

--------------------------------

I hope this can help you solve more than half of the problems in every contest, 
from someone who has just got acquainted with Python.

Most of my practice on LeetCode is limited to contest, so I may not be aware of better examples. 
I think this article could be improved with more appropriate LeetCode problems that better illustrates 
the concept behind the keyword I am introducing.