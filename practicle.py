# Node class
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

# Inorder Traversal Function
def inorder(root):
    if root:
        inorder(root.left)        # Step 1: Left
        print(root.data, end=" ") # Step 2: Root
        inorder(root.right)       # Step 3: Right

# Creating Binary Tree
root = Node(4)
root.left = Node(2)
root.right = Node(5)
root.left.left = Node(1)
root.left.right = Node(3)

# Calling function
print("Inorder Traversal:")
inorder(root)