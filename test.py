class Solution:
    def search(self, nums, target: int) -> int:
        self.nums = nums
        self.target = target
        return self.binarySearch(0, len(self.nums) - 1)

    def binarySearch(self, left: int, right: int) -> int:
        if left > right:
            return -1
        mid = (left + right) // 2
        m = self.nums[mid]
        if m == self.target:
            return mid
        if self.nums[0] <= self.target < m or m < self.nums[0] <= self.target or self.target < m < self.nums[0]:
            return self.binarySearch(left, mid - 1)
        if self.nums[0] <= m < self.target or self.target < self.nums[0] <= m or m < self.target < self.nums[0]:
            return self.binarySearch(mid + 1, right)
        return -1


print(Solution().search([1,3,5], 1))
