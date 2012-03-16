/* red-black tree */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>


typedef int T;                  /* type of item to be stored */
#define compLT(a,b) (a < b)
#define compEQ(a,b) (a == b)

/* Red-Black tree description */
typedef enum { BLACK, RED } nodeColor;

typedef struct Node_ {
    struct Node_ *left;         /* left child */
    struct Node_ *right;        /* right child */
    struct Node_ *parent;       /* parent */
    nodeColor color;            /* node color (BLACK, RED) */
    T node_data;                /* data stored in node */
} Node;

#define NIL &sentinel           /* all leafs are sentinels */
Node sentinel = { NIL, NIL, 0, BLACK, 0};



//=============================================================================
// The RB-Tree management functions.
//
// Note: The internal functions are declared as static
//=============================================================================


//-----------------------------------------------------------------------------
/// rotate node x to left
//-----------------------------------------------------------------------------
static void rotateLeft(Node *x, Node**p_root)
{
    Node *y = x->right;

    /* establish x->right link */
    x->right = y->left;
    if (y->left != NIL) y->left->parent = x;

    /* establish y->parent link */
    if (y != NIL) y->parent = x->parent;
    if (x->parent)
    {
        if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;
    }
    else
    {
        *p_root = y;
    }

    /* link x and y */
    y->left = x;
    if (x != NIL) x->parent = y;
}

//-----------------------------------------------------------------------------
/// rotate node x to right
//-----------------------------------------------------------------------------
static void rotateRight(Node *x, Node**p_root)
{
    Node *y = x->left;

    /* establish x->left link */
    x->left = y->right;
    if (y->right != NIL) y->right->parent = x;

    /* establish y->parent link */
    if (y != NIL) y->parent = x->parent;
    if (x->parent)
    {
        if (x == x->parent->right)
            x->parent->right = y;
        else
            x->parent->left = y;
    }
    else
    {
        *p_root = y;
    }

    /* link x and y */
    y->right = x;
    if (x != NIL) x->parent = y;
}

//-----------------------------------------------------------------------------
/// maintain Red-Black tree balance after inserting node x 
//-----------------------------------------------------------------------------
static void insertFixup(Node *x, Node**p_root)
{
    /* check Red-Black properties */
    while (x != *p_root && x->parent->color == RED)
    {
        /* we have a violation */
        if (x->parent == x->parent->parent->left)
        {
            Node *y = x->parent->parent->right;
            if (y->color == RED)
            {

                /* uncle is RED */
                x->parent->color = BLACK;
                y->color = BLACK;
                x->parent->parent->color = RED;
                x = x->parent->parent;
            }
            else
            {

                /* uncle is BLACK */
                if (x == x->parent->right)
                {
                    /* make x a left child */
                    x = x->parent;
                    rotateLeft(x, p_root);
                }

                /* recolor and rotate */
                x->parent->color = BLACK;
                x->parent->parent->color = RED;
                rotateRight(x->parent->parent, p_root);
            }
        }
        else
        {

            /* mirror image of above code */
            Node *y = x->parent->parent->left;
            if (y->color == RED)
            {

                /* uncle is RED */
                x->parent->color = BLACK;
                y->color = BLACK;
                x->parent->parent->color = RED;
                x = x->parent->parent;
            }
            else
            {

                /* uncle is BLACK */
                if (x == x->parent->left)
                {
                    x = x->parent;
                    rotateRight(x, p_root);
                }
                x->parent->color = BLACK;
                x->parent->parent->color = RED;
                rotateLeft(x->parent->parent, p_root);
            }
        }
    }
    (*p_root)->color = BLACK;
}

//-----------------------------------------------------------------------------
/// insert new node in the tree as a child of given parent (after find was made)
/// \param[in] x      - new node to insert
/// \param[in] parent - the parent of new node. This ptr is calculated by findNode function
/// \param[in] p_root - the root of the tree
//-----------------------------------------------------------------------------
void insertNodeToParent(Node *x, Node*parent, Node**p_root)
{
    x->parent = parent;
    x->left = NIL;
    x->right = NIL;
    x->color = RED;
    
    /* insert node in tree */
    if(parent) {
        if(compLT(x->node_data, parent->node_data))
            parent->left = x;
        else
            parent->right = x;
    } else {
        *p_root = x;
    }

    insertFixup(x, p_root);
    return;
}

//-----------------------------------------------------------------------------
/// maintain Red-Black tree balance after deleting node x
//-----------------------------------------------------------------------------
static void deleteFixup(Node *x, Node *x_par, Node**p_root)
{
    while (x != *p_root && x->color == BLACK) {
        if (x == x_par->left) {
            Node *w = x_par->right;
            if (w->color == RED) {
                w->color = BLACK;
                x_par->color = RED;
                rotateLeft(x_par, p_root);
                w = x_par->right;
            }
            if (w->left->color == BLACK && w->right->color == BLACK) {
                w->color = RED;
                x = x_par;
                x_par = x->parent;
            } else {
                if (w->right->color == BLACK) {
                    w->left->color = BLACK;
                    w->color = RED;
                    rotateRight(w, p_root);
                    w = x_par->right;
                }
                w->color = x_par->color;
                x_par->color = BLACK;
                w->right->color = BLACK;
                rotateLeft (x_par, p_root);
                x = *p_root;
                x_par = NULL;
            }
        } else {
            Node *w = x_par->left;
            if (w->color == RED) {
                w->color = BLACK;
                x_par->color = RED;
                rotateRight (x_par, p_root);
                w = x_par->left;
            }
            if (w->right->color == BLACK && w->left->color == BLACK) {
                w->color = RED;
                x = x_par;
                x_par = x->parent;

            } else {
                if (w->left->color == BLACK) {
                    w->right->color = BLACK;
                    w->color = RED;
                    rotateLeft (w, p_root);
                    w = x_par->left;
                }
                w->color = x_par->color;
                x_par->color = BLACK;
                w->left->color = BLACK;
                rotateRight (x_par, p_root);
                x = *p_root;
                x_par = NULL;
            }
        }
    }
    x->color = BLACK;
}

//-----------------------------------------------------------------------------
/// detach node from tree
/// \param[in] z -- the node to detach
/// \param[in] p_root - the root of the tree
//-----------------------------------------------------------------------------
void detachNode(Node *z, Node**p_root)
{
    Node *x, *y, *x_par;
    int f;

    if(!*p_root)return;

    if (!z || z == NIL) return;

    if (z->left == NIL || z->right == NIL) {
        /* y has a NIL node as a child */
        y = z;
    } else {
        /* find tree successor with a NIL node as a child */
        y = z->right;
        while (y->left != NIL) y = y->left;
    }

    /* x is y's only child */
    if (y->left != NIL)
        x = y->left;
    else
        x = y->right;

    /* remove y from the parent chain */
    x_par = y->parent;
    if(x!=NIL) x->parent = x_par;
        
    if (y->parent)
        if (y == y->parent->left)
            y->parent->left = x;
        else
            y->parent->right = x;
    else
        *p_root = x;

    f = (y->color == BLACK);
    
    if (y != z)
    {   /* replace z with y */ 
        y->color = z->color ;
        y->left = z->left;
        y->right = z->right;
        y->parent = z->parent;

        if (y->parent)
            if (z == y->parent->left)
                y->parent->left = y;
            else
                y->parent->right = y;
        else
            *p_root = y;

        if(y->left != NIL)
            y->left->parent = y;

        if(y->right != NIL)
            y->right->parent = y;

        if(x_par==z) x_par=y;
    }

    if(f)
        deleteFixup (x, x_par, p_root);

}

//-----------------------------------------------------------------------------
///  find node containing data and place to insert of not found 
/// \param[in] d               -- data to find
/// \param[in] p_root          -- refers to the tree root
/// \param[out]p_insert_parent -- where to insert the new node if not found:
///                               - *p_insert_parent := parent
///                               - if node found, this ptr is unchanged!
///                               - if no parent (empty tree) 0 is returned in this ptr
///                               this value should be passed to insertNodeToParent() function 
/// \return ptr to node found, or 0 if not found.
//-----------------------------------------------------------------------------
Node *findNode(T d, Node**p_root, Node**p_insert_parent)
{
    if(*p_root)
    {   Node *current = *p_root;
        Node *parent = 0;
        while(current != NIL)
            if(compEQ(d, current->node_data))
                return (current);
            else
            {   parent=current;
                current = compLT (d, current->node_data) ? current->left
                                                         : current->right;
            }
    
        if(p_insert_parent) *p_insert_parent = parent;
        return(0);
    }
    else
    {   if(p_insert_parent) *p_insert_parent = 0;
        return(0);
    }
}

//==============================================================================
//  Test code
//==============================================================================
Node *root = NIL;               /* root of Red-Black tree */


void main(int argc, char **argv) {
    int a, maxnum, ct;
    Node *t,*ins_pos;

    /* command-line:
     *
     *   rbt maxnum
     *
     *   rbt 2000
     *       process 2000 records
     *
     */

    maxnum = atoi(argv[1]);

    for (ct = maxnum; ct; ct--) {
        a = rand() % 9 + 1;
        if ((t = findNode(a, &root, &ins_pos)) != NULL) {
            detachNode(t, &root);
	        free(t);
        } else {
            if ((t = malloc (sizeof(Node))) == 0) {
                printf ("insufficient memory (insertNode)\n");  
                exit(1);
            }
            t->node_data = a;
            insertNodeToParent(t, ins_pos, &root);
        }
    }
}
