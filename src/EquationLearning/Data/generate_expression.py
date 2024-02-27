import random


class Node:
    def __init__(self, nodeOp: str, nodeType: str, parent):
        self.nodeOp = nodeOp
        self.nodeType = nodeType
        self.children = []
        self.parent = parent

    def setChildren(self, child):
        self.children.append(child)
        
        
class GenExpression:
    
    def __init__(self, max_tokens, unary_ops, max_nest):
        self.max_tokens = max_tokens
        self.unary_ops = unary_ops
        self.un_ops_orig = self.unary_ops.copy()
        self.max_nest = max_nest

    def addNode(self, prev_node, nt, nest_level, nest_bin=0):
        if nt == 0:
            type_node = random.choice(['unary', 'binary'])
        else:
            if nt + 1 > self.max_tokens:
                type_node = random.choice(['leaf'])
            elif nest_level == self.max_nest or len(self.unary_ops) == 0:
                type_node = random.choice(['binary', 'leaf'])
            else:
                type_node = random.choices(['unary', 'binary', 'leaf'], weights=[2, 1, 1], k=1)[0]
            if nest_bin == 2:
                type_node = 'leaf'
    
        if type_node == 'binary':
            bin_op = random.choice(['add', 'mul', 'div'])
            # Create current node
            currentNode = Node(nodeOp=bin_op, nodeType='binary', parent=prev_node)
            init_nest_bin = nest_bin
            nest_bin += 1
            nt += 1
            # Create children nodes
            child1, nt, nest_level, _ = self.addNode(prev_node=currentNode, nt=nt, nest_level=nest_level, nest_bin=nest_bin)
            child2, nt, nest_level, _ = self.addNode(prev_node=currentNode, nt=nt, nest_level=nest_level, nest_bin=nest_bin)
            nest_bin = init_nest_bin
    
        elif type_node == 'unary':
            unary_ops = self.unary_ops.copy()
            if nest_level >= 1:
                # Find what's the previous unary operator
                parent = prev_node
                while parent.nodeType != 'unary':
                    parent = prev_node.parent
                # Prevent forbidden operations
                ops_to_eliminate = []
                if parent.nodeOp == 'abs':
                    ops_to_eliminate = ["sqrt", "pow2", "pow4"]
                elif parent.nodeOp in ['exp', 'tan', 'ln']:
                    ops_to_eliminate = ['exp', 'sinh', 'cosh', 'tanh', 'tan', 'ln', 'pow3', 'pow4', 'pow5']
                elif parent.nodeOp in ['sinh', 'cosh', 'tanh']:
                    ops_to_eliminate = ['exp', 'sinh', 'cosh', 'tanh', 'tan', 'ln', 'pow2', 'pow3', 'pow4', 'pow5']
                elif parent.nodeOp in ['pow2', 'pow3', 'pow4', 'pow5']:
                    ops_to_eliminate = ['pow2', 'pow3', 'pow4', 'pow5', 'exp', 'sinh', 'cosh', 'tanh']
                elif parent.nodeOp in ['sin', 'cos', 'tan']:
                    ops_to_eliminate = ['sin', 'cos', 'tan']
                elif parent.nodeOp in ['asin', 'acos', 'atan']:
                    ops_to_eliminate = ['asin', 'acos', 'atan']

                unary_ops = [op for op in self.unary_ops if op not in ops_to_eliminate]
    
            if len(unary_ops) > 0:
                un_op = random.choice(unary_ops)
                init_nest_level = nest_level
                nest_level += 1
                nt += 1
                # Create current node
                currentNode = Node(nodeOp=un_op, nodeType='unary', parent=prev_node)
                # Create child node
                child1, nt, _, nest_bin = self.addNode(prev_node=currentNode, nt=nt, nest_level=nest_level, nest_bin=nest_bin)
                nest_level = init_nest_level
            else:
                if len(prev_node.children) > 0:  # If this node has a sibling, analyze it
                    if prev_node.children[0].nodeOp == 'x_1':
                        leaf = '1'
                    else:
                        leaf = 'x_1'
                else:
                    if prev_node.nodeType == 'unary':
                        leaf = 'x_1'
                    else:
                        leaf = random.choice(['x_1', '1'])
                # Create current node
                currentNode = Node(nodeOp=leaf, nodeType='leaf', parent=prev_node)
    
        else:  # leaf
            if len(prev_node.children) > 0:  # If this node has a sibling, analyze it
                if prev_node.children[0].nodeOp == 'x_1':
                    leaf = '1'
                else:
                    leaf = 'x_1'
            else:
                if prev_node.nodeType == 'unary':
                    leaf = 'x_1'
                else:
                    leaf = random.choice(['x_1', '1'])
            # Create current node
            currentNode = Node(nodeOp=leaf, nodeType='leaf', parent=prev_node)
    
        if prev_node is not None:
            prev_node.setChildren(currentNode)

        self.unary_ops = [op for op in self.unary_ops if op != currentNode.nodeOp]
    
        return currentNode, nt, nest_level, nest_bin    
    
    def generate_expr_tree(self):
        prev_node = None
        init_nt = 0
        nodeOrigin = self.addNode(prev_node=prev_node, nt=init_nt, nest_level=0)
        fexpr = self.tree_to_str(nodeOrigin[0], [])
        while not any([un_op in str(fexpr) for un_op in self.un_ops_orig]):
            fexpr = self.addNode(prev_node=prev_node, nt=init_nt, nest_level=0)
            fexpr = self.tree_to_str(fexpr[0], [])
        return fexpr
        
    def tree_to_str(self, node, strng):
        if len(node.children) == 0:
            strng.append(node.nodeOp)
            return strng
        else:
            strng.append(node.nodeOp)
            strng = self.tree_to_str(node.children[0], strng)
            if len(node.children) == 2:
                strng = self.tree_to_str(node.children[1], strng)
            return strng


if __name__ == '__main__':
    m_tokens = 5
    un_ops = ['sin', 'cos', 'pow2']
    gen = GenExpression(max_tokens=m_tokens, unary_ops=un_ops, max_nest=2)
    expr = gen.generate_expr_tree()
    print(expr)
