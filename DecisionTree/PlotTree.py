from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt


# 绘制决策树
class PlotTree:

    def __init__(self, decision_tree, font_path='/System/Library/Fonts/Hiragino Sans GB.ttc'):
        self.font_path = font_path  # 中文字体路径
        self.decision_tree = decision_tree
        self.totalW = float(self._get_leaf_num(self.decision_tree))  # 获取决策树叶结点数目
        self.totalD = float(self._get_tree_depth(self.decision_tree))  # 获取决策树层数
        self.xOff = -0.5 / self.totalW
        self.yOff = 1

    @classmethod
    def _get_leaf_num(cls, decision_tree):
        leaf_num = 0
        for key, value in decision_tree.items():
            if isinstance(value, dict):
                leaf_num += cls._get_leaf_num(value)
            else:
                leaf_num += 1
        return leaf_num

    @classmethod
    def _get_tree_depth(cls, decision_tree):
        max_depth = 0
        for key, value in decision_tree.items():
            if isinstance(value, dict):
                this_depth = 1 + cls._get_leaf_num(value)
            else:
                this_depth = 1
            if this_depth > max_depth:
                max_depth = this_depth
        return max_depth

    def plot_mid_text(self, center_pt, parent_pt, txt):
        x_mid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
        y_mid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]
        self.ax1.text(x_mid, y_mid, txt, va="center", ha="center", rotation=20,
                      FontProperties=FontProperties(fname=self.font_path, size=13))

    def plot_node(self, node_txt, center_pt, parent_pt, node_type):
        arrow_args = dict(arrowstyle="<-")
        self.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                          xytext=center_pt, textcoords='axes fraction',
                          va="center", ha="center", bbox=node_type, arrowprops=arrow_args,
                          FontProperties=FontProperties(fname=self.font_path, size=15))

    def plot_tree(self, tree, parent_pt, node_txt):

        decision_node = dict(boxstyle="sawtooth", fc="0.8")
        leaf_node = dict(boxstyle="round4", fc="0.8")
        leaf_num = self._get_leaf_num(tree)
        first_str = next(iter(tree))
        center_pt = (self.xOff + (1.0 + float(leaf_num)) / 2.0 / self.totalW, self.yOff)
        self.plot_mid_text(center_pt, parent_pt, node_txt)
        self.plot_node(first_str, center_pt, parent_pt, decision_node)
        second_dict = tree[first_str]
        self.yOff = self.yOff - 2.0 / self.totalD
        for key in second_dict.keys():
            if isinstance(second_dict[key], dict):
                self.plot_tree(second_dict[key], center_pt, str(key))
            else:
                self.xOff = self.xOff + 1.0 / self.totalW
                self.plot_node(second_dict[key], (self.xOff, self.yOff), center_pt, leaf_node)
                self.plot_mid_text((self.xOff, self.yOff), center_pt, str(key))
        self.yOff = self.yOff + 2 / self.totalD

    def plot_img(self):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        self.ax1 = plt.subplot(111, frameon=False, xticks=[], yticks=[])
        self.plot_tree(self.decision_tree, (0.5, 1.0), '')
        plt.show()

    def __call__(self, *args, **kwargs):
        self.plot_img()

    @classmethod
    def plot(cls, decision_tree):
        plot_obj = cls(decision_tree)
        plot_obj.plot_img()
