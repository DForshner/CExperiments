/*
 ============================================================================
 Name: SimpleBinarySearchTree.c
 Description: Simple binary tree.
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*
 * Binary Tree Structs
 */

/* Node of binary tree */
struct btree_node {
    int data;
    struct btree_node *left;
	struct btree_node *right;
};

/* Binary tree */
struct btree_tree {
	struct btree_node *root;
	int count;
};

/*
 * Binary Tree Operations
 */

/*
 * Creation
 * Parameters: binary tree to create
 */
struct btree_tree *btree_create(void) {

	struct btree_tree *tree = malloc (sizeof *tree);

    if (tree == NULL)
		return NULL;

	tree->root = NULL;
	tree->count = 0;

	return tree;
}

/*
 * Search
 * Parameters: binary tree, item to search for
 */
int btree_search(const struct btree_tree *tree, int item) {
	const struct btree_node *node;
	
	/* Require that tree points to a non NULL pointer. */
	assert(tree != NULL);
	
	node = tree->root;
	
	for (;;) {
		/* End: out of nodes */
		if (node == NULL)
			return 0;
		/* End: Found item */
		else if (item == node->data)
			return 1;
		/* If the item is larger go right. */
		else if (item > node->data)
			node = node->right;
		/* If the item is smaller go left. */
		else
			node = node->left; 
	}
	
}

/*
 * Insert item into tree
 * Parameters: binary tree, item to insert
 */
int btree_insert(struct btree_tree *tree, int item) {
	struct btree_node *node;

	/* Points to pointer that was followed to arrive at node. */
	struct btree_node **p_to_node;
	
	/* Verify the tree is not null. */
	assert(tree != NULL);
	
	node = tree->root;
	p_to_node = &tree->root;

	for (;;)
	{
		/* If the item wasn't found before we reached a node that is NULL */
		if (node == NULL)
		{
			/* Replace null pointer followed to get here with a node containing the item. */
			node = *p_to_node = malloc(sizeof *node);

			/* Check that the memory allocation succeeded */
			if (node != NULL)
			{
				node->data = item;
				node->left = node->right = NULL;
				tree->count++;
				return 1;
			}
			else
				return 0;
		}
		/* If item already exists in tree. */
		else if (item == node->data)
		{
			return 2;
		}
		/* If item is larger go right. */
		else if (item > node->data)
		{
			p_to_node = &node->right;
			node = node->right;
		}
		/* If item is smaller go left. */
		else
		{
			p_to_node = &node->left;
			node = node->left;
		}
	}
}

/*
 * Delete item from tree
 * Parameters: binary tree, item to delete
 */
int btree_delete(struct btree_tree *tree, int item) {
	struct btree_node *node;

	/* Points to the pointer that was followed to arrive at the current node. */
	struct btree_node **p_to_node;

	/* Verify tree is not null. */
	assert(tree != NULL);
	
	/* Set current node to the tree root */
	p_to_node = &tree->root;
	node = tree->root;
	
	/*
	 * Search the tree for the item
	 */
	for (;;)
	{
        /* Item not found */
		if (node == NULL)
        {
			return 0;
        }
		/* if the item was found */
		else if (item == node->data)
		{
			break;
		}
		/* if item is larger than the current node search right */
		else if (item > node->data)
		{
			p_to_node = &node->right;
			node = node->right;
		}
		/* if item is smaller than current node search left */
		else
		{
			p_to_node = &node->left;
			node = node->left;
		}
	}
	
	/*
	 * Remove the item from the tree
	 */

	/* if the current node has no right child. */
	if (node->right == NULL)
	{
		/* The left could be a child node or NULL. */
		*p_to_node = node->left;
	}
	else
	{
		struct btree_node *rightChild = node->right;
		
		/* if the current node has a right child and this right child's left child is null. */
		if (rightChild->left == NULL)
		{
			/* The left could be a child node or NULL. */
			rightChild->left = node->left;
			*p_to_node = rightChild;
		}
		/* if the current node has a right child and this right child has a left child. */
		else
		{
			struct btree_node *rightChildsLeft = rightChild->left;
			
			/*
			 * Search down the left side until there is an node with no left child.
			 * This node is the smallest value in the tree greater than the item being deleted.
			 */
			while (rightChildsLeft->left != NULL)
			{
				rightChild = rightChildsLeft;
				rightChildsLeft = rightChild->left;
			}
			
			/* Remove the leftmost (smallest) node from the tree */
			rightChild->left = rightChildsLeft->right;

			/* Replace the item with the leftmost (smallest) node */
			rightChildsLeft->left = node->left;
			rightChildsLeft->right = node->right;
			*p_to_node = rightChildsLeft;
		}
	}
	
	tree->count--;
	free(node);
	return 1;
}

/*
 * Display the results of searching for an item before and after removing it from the tree.
 * Parameters: binary tree, item to remove
 */
void test_remove(struct btree_tree *tree, int item) {
	  printf("Before remove item %i - Search resulted in %i.\n", item, btree_search(tree, item));
	  btree_delete(tree, item);
	  printf("After remove item %i - Search resulted in %i.\n\n", item, btree_search(tree, item));
}

/*
 * Main - Try inserting / searching / deleting some nodes from a tree.
 */
int main(void) {
  printf("-- Simple Binary Search Tree --\n");

  /* Populate a tree */
  struct btree_tree *test = btree_create();
  btree_insert(test, 10);
  btree_insert(test, 3);
  btree_insert(test, 15);
  btree_insert(test, 9);
  btree_insert(test, 14);
  btree_insert(test, 20);
  btree_insert(test, 12);
  btree_insert(test, 21);
  btree_insert(test, 16);
  btree_insert(test, 17);
  btree_insert(test, 18);
  btree_insert(test, 11);
  btree_insert(test, 13);
  btree_insert(test, 2);

  printf("\nThere are now %i items in the tree.\n\n", test->count);

  int item;

  /* Remove an item with no right child */
  item = 2;
  test_remove(test, item);

  /* Remove an item with a right child that has no left child */
  item = 12;
  test_remove(test, item);

  /* Remove an item with a right child that has a left child */
  item = 20;
  test_remove(test, item);

  printf("There are now %i items in the tree.", test->count);

  return EXIT_SUCCESS;
}
