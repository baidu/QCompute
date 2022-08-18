# 量子魔方游戏

*版权所有 (c) 2022 百度量子计算研究所，保留所有权利。*

在前面的教程中，我们介绍了 CHSH 游戏，并展示了在该游戏的量子最优策略中，如何通过引入量子纠缠并制定合理的策略来提高玩家的获胜概率。然而，根据 Tsirelson 边界 [1]，该游戏中最优策略也只能实现约 $85\%$ 的胜率。本教程中，我们将介绍另一种有趣而又经典的量子非局域游戏——量子魔方游戏（Mermin-Peres magic square game）[2,3]。在魔方游戏中，玩家通过量子策略可以实现 $100\%$ 的胜率，而通过经典策略则最多只能以 $88.89\%$ 的概率取胜。

接下来，我们将对魔方游戏的游戏规则及其对应的经典最优策略和量子最优策略进行介绍，并借助 QNET 对该游戏的量子最优策略进行仿真验证，以见证量子在魔方游戏中所体现出的惊人优势。

## 游戏介绍

### 1. 游戏规则

在魔方游戏中，共有三位角色：分别为两名属于合作关系的玩家 Alice 和 Bob，以及一名裁判 Referee。

其中，裁判将在本地保存一张 $3 * 3$ 的表格，玩家 Alice 和 Bob 需要根据裁判指定的行号或列号分别对表格中的某一行或某一列进行填充，每一格中可填入的元素取值为 $+1$ 或 $-1$。最后，由裁判对结果进行分析并判断游戏胜负。

该游戏的具体游戏规则如下：

1. 游戏开始时，由裁判随机抽取 2 个问题 $x, y \in \{0, 1, 2\}$，分别对应两位玩家需要填充的行号和列号；
2. Alice 和 Bob 需要根据裁判的提问分别对表格的第 $x$ 行和第 $y$ 列使用数字 $+1$ 或 $-1$ 进行填充，并将他们的填充结果作为回答返回给裁判；
3. 裁判收到 Alice 和 Bob 的回答之后，将对应的元素填入他所持有的表格中。如果 Alice 所填充的一行元素乘积为 $+1$，Bob 所填充的一列元素乘积为 $-1$，且他们在行列相交处所填的元素一致，则宣布玩家获胜；否则宣布玩家失败。（图 1 展示了当 $x=1, y=2$ 时玩家获胜的一种填表方式。）

**注意**：在游戏开始前，两位玩家之间可以共同建立一个合理的游戏策略，以此来提高他们的获胜概率。但是一旦游戏开始，则玩家之间不允许进行任何通信。

![图 1：$x=1, y=2$ 时一种获胜的填表方式](./figures/magic_square_game-example.png "图 1：$x=1, y=2$ 时一种获胜的填表方式")

### 2. 经典最优策略

为尽可能提高获胜概率，Alice 和 Bob 可以预先对表格中各元素的填充方案进行协商。经过尝试后会发现，他们最多只能确定表格中 8 个元素的取值，不论表格中最后一个元素如何取值，都无法满足获胜条件。

一种可能的经典策略如图 2 所示。不难发现，采用这种填表方式，在 $x=2, y=2$ 的情况下，Alice 和 Bob 总是无法对行列交叉点处的元素达成一致。若填入 $-1$，则 Bob 无法满足获胜要求；反之，若填入 $+1$，则 Alice 一行元素的乘积不为 $+1$，同样无法满足获胜要求。在该策略下，玩家最多只能达到 $\frac{8}{9}$ 的获胜概率，总会存在一种情况使得玩家输掉比赛。另一方面，我们可以验证任何经典策略都无法获得超过 $\frac{8}{9}$ 的胜率，即图 2 的策略为经典最优策略。

![图 2：一种可能的最优经典策略](./figures/magic_square_game-optimal_classical_strategy.png "图 2：一种可能的最优经典策略")

### 3. 量子最优策略

接下来我们将对魔方游戏的量子最优策略进行介绍。在该策略中，通过引入两对共享的纠缠对，玩家可以保证以 $100\%$ 的概率获得游戏的胜利。

魔方游戏的量子最优策略如下：

1. Alice 和 Bob 在游戏开始之前，需要提前共享两对贝尔态：$|\Phi^+\rangle_{A_1B_1}=\frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)$，$|\Phi^+\rangle_{A_2B_2}=\frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)$，其中 $A_1, A_2$ 和 $B_1, B_2$ 分别为 Alice 和 Bob 所持有的粒子；
2. 当裁判分别向 Alice 和 Bob 提问，他们将根据裁判给出的问题分别对他们所持有的粒子 $A_1A_2$ ($B_1B_2$) 执行对应的操作，并根据对所持粒子的测量结果作出对应的回答。具体操作和回答方式如下表所示。

|   问题   |                                   玩家操作                                    |     测量结果 "00"      |   测量结果 "01"   |   测量结果 "10"   |  测量结果 "11"   |
|:------:|:-------------------------------------------------------------------------:|:------------------:|:-------------:|:-------------:|:------------:|
| $x=0$  |                              对两个粒子执行 $Z$ 基测量                              |    (+1, +1, +1)    | (-1, +1, -1)  | (+1, -1, -1)  | (-1, -1, +1) |
| $x=1$  |                              对两个粒子执行 $X$ 基测量                              |    (+1, +1, +1)    | (+1, -1, -1)  | (-1, +1, -1)  | (-1, -1, +1) |
| $x=2$  | 对两个粒子先依次作用 $Z_{A_1}, Z_{A_2}, CZ_{A_1 A_2}, H_{A_1}, H_{A_2}$，再执行 $Z$ 基测量 |    (+1, +1, +1)    | (+1, -1, -1)  | (-1, +1, -1)  | (-1, -1, +1) |
| $y=0$  |                     对第一个粒子执行 $X$ 基测量，对第二个粒子执行 $Z$ 基测量                     |    (+1, +1, -1)    | (-1, +1,  +1) | (+1, -1,  +1) | (-1, -1,-1)  |
| $y=1$  |                     对第一个粒子执行 $Z$ 基测量，对第二个粒子执行 $X$ 基测量                     |    (+1, +1, -1)    | (+1, -1,  +1) | (-1, +1,  +1) | (-1, -1,-1)  |
| $y=2$  |             对两个粒子先依次作用 $CNOT_{B_1 B_2}, H_{B_1}$，再执行 $Z$ 基测量              |    (+1, +1, -1)    | (-1, +1,  +1) | (+1, -1,  +1) | (-1, -1,-1)  |

关于上述策略的最优性证明可参见文献 [2, 3]。

接下来，我们将使用 QNET 对魔方游戏及其量子最优策略进行仿真，以验证在魔方游戏中采用量子策略的优越性。

## 协议实现

我们在 QPU 模块中提供了采用量子最优策略的魔方游戏协议 —— ``MagicSquareGame`` 类，其中包含 4 个子协议，分别对应描述魔方游戏中四种不同角色的行为：负责制备和分发纠缠的纠缠源（``Source``），根据提问填充表格第 $x$ 行的玩家（``Player1``），填充表格第 $y$ 列的玩家（``Player2``），以及向玩家分发问题并回收答案、最后根据表格填充结果判断游戏胜负的裁判（``Referee``）。

**注意**：由于需要对该游戏的获胜概率进行统计，需要重复进行多轮次的游戏。协议中每开启新一轮的游戏，需要生成一个新的量子电路，并使用其对应的索引同步每一轮次各方的行为。


```python
class MagicSquareGame(Protocol):

    def __init__(self, name=None):
        super().__init__(name)
        self.role = None

    class Message(ClassicalMessage):

        def __init__(self, src: "Node", dst: "Node", protocol: type, data: Dict):
            super().__init__(src, dst, protocol, data)

        @unique
        class Type(Enum):

            ENT_REQUEST = "Entanglement request"
            READY = "Ready"
            QUESTION = "Question from the referee"
            ANSWER = "Answer from the player"

    def start(self, **kwargs) -> None:
        role = kwargs['role']
        # 根据角色实例化对应子协议
        self.role = getattr(MagicSquareGame, role)(self)  
        # 启动子协议
        self.role.start(**kwargs)  

    def receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        # 由对应的子协议接收经典消息
        self.role.receive_classical_msg(msg)  

    def receive_quantum_msg(self, msg: "QuantumMsg", **kwargs) -> None:
        # 存储接收到的量子比特
        self.node.qreg.store_qubit(msg.data, kwargs['src'])  
        # 同步对应的量子电路
        self.node.qreg.circuit_index = msg.index  
        # 由对应的子协议做进一步的操作
        self.role.receive_quantum_msg()  

    def estimate_statistics(self, results: List[Dict]) -> None:
        # 确保该方法只能由裁判进行调用
        assert type(self.role).__name__ == "Referee",\
            f"The role of {type(self.role).__name__} has no right to calculate the winning probability of the game!"
        # 调用相应子协议中的该方法
        self.role.estimate_statistics(results)  
```

### 1. 纠缠源（``Source``）

纠缠源负责在游戏开始前为两位玩家分发纠缠。当收到 Alice 从经典信道发来的纠缠请求信息时，它将通过量子寄存器的 ``create_circuit`` 方法，生成一个新的量子电路并添加到 ``Network`` 的 ``circuits`` 列表中。然后，它在本地制备两对贝尔态 $| \Phi^+ \rangle = \tfrac{1}{\sqrt{2}} (|00 \rangle + |11 \rangle)$，并通过量子信道将每对纠缠对中的两个纠缠粒子依次分发给 Alice 和 Bob。


```python
class MagicSquareGame(Protocol):
    ...
    class Source(SubProtocol):

        def __init__(self, super_protocol: Protocol):
            super().__init__(super_protocol)

        def start(self, **kwargs) -> None:
            pass

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            if msg.data['type'] == MagicSquareGame.Message.Type.ENT_REQUEST:
                # 生成一个新的量子电路
                self.node.qreg.create_circuit(f"MagicSquareGame_")  
                # 在本地生成两对纠缠对
                self.node.qreg.h(0)
                self.node.qreg.cnot([0, 1])
                self.node.qreg.h(2)
                self.node.qreg.cnot([2, 3])

                # 对生成的纠缠粒子进行分发，使用 priority 参数规定相同动作的先后行为（priority 的值越小，事件优先级越高）
                self.node.send_quantum_msg(dst=msg.src, qreg_address=0, priority=0)
                self.node.send_quantum_msg(dst=msg.src, qreg_address=2, priority=1)
                self.node.send_quantum_msg(dst=msg.data['peer'], qreg_address=1, priority=0)
                self.node.send_quantum_msg(dst=msg.data['peer'], qreg_address=3, priority=1)
```

### 2. 玩家 1（``Player1``）

在该协议中，玩家 1 的行为包括：

1. 游戏正式开始前，通过 ``prepare_for_game`` 方法，向纠缠源发送纠缠请求消息，请求与另一位玩家共享纠缠；
2. 确认收到两份纠缠粒子后，向裁判发送 ``READY`` 信息，开始魔方游戏；
3. 收到裁判的问题 $x$（需要填入元素的行号）后，对她所持的两个粒子执行对应的操作并测量；
4. 将测量结果作为回答返回给裁判。

**注意**：按照前面介绍的最优量子策略，测量结果唯一确定了最终的回答方式，所以这里玩家双方只需要将他们的测量结果告知裁判，然后由裁判按照他们事先约定的策略为对应的行、列进行数字填充并判断胜负即可。


```python
class MagicSquareGame(Protocol):
    ...
    class Player1(SubProtocol):

        def __init__(self, super_protocol: Protocol):
            super().__init__(super_protocol)
            self.peer = None
            self.ent_source = None
            self.referee = None
            self.rounds = None
            self.current_round = 0

        def start(self, **kwargs) -> None:
            # 另一位游戏玩家
            self.peer = kwargs['peer']  
            # 纠缠源
            self.ent_source = kwargs['ent_source']  
            # 游戏裁判
            self.referee = kwargs['referee']  
            # 游戏总轮数
            self.rounds = kwargs['rounds']  
            # 请求纠缠资源，准备开始游戏
            self.prepare_for_game()  

        def prepare_for_game(self) -> None:
            # 即将开始的游戏的轮数
            self.current_round += 1  
            # 向纠缠源发起纠缠请求
            self.request_entanglement()  

        def request_entanglement(self) -> None:
            # 生成一则纠缠请求消息
            ent_request_msg = MagicSquareGame.Message(
                src=self.node, dst=self.ent_source, protocol=MagicSquareGame,
                data={'type': MagicSquareGame.Message.Type.ENT_REQUEST, 'peer': self.peer}
            )
            # 向纠缠源发送纠缠请求
            self.node.send_classical_msg(dst=self.ent_source, msg=ent_request_msg)  

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            if msg.data['type'] == MagicSquareGame.Message.Type.QUESTION:
                # 记录需要填入元素的行号
                row = msg.data['question']  

                # 若需要填入第 0 行，对所持粒子执行 Z 基测量
                if row == 0:
                    self.node.qreg.measure(0, basis="z")
                    self.node.qreg.measure(1, basis="z")
                    self.node.qreg.circuit.name += "r0"  # 记录玩家 1 需要填写的行号

                # 若需要填入第 1 行，对所持粒子执行 X 基测量
                elif row == 1:
                    self.node.qreg.measure(0, basis="x")
                    self.node.qreg.measure(1, basis="x")
                    self.node.qreg.circuit.name += "r1"  # 记录玩家 1 需要填写的行号

                # 若需要填入第 2 行，对所持粒子先执行各种量子门操作，再执行 Z 基测量
                elif row == 2:
                    self.node.qreg.z(0)
                    self.node.qreg.z(1)
                    self.node.qreg.cz([0, 1])
                    self.node.qreg.h(0)
                    self.node.qreg.h(1)
                    self.node.qreg.measure(0, basis="z")
                    self.node.qreg.measure(1, basis="z")
                    self.node.qreg.circuit.name += "r2"  # 记录该电路所对应玩家 1 需要填写的行号

                # 将答复消息发送给裁判
                answer_msg = MagicSquareGame.Message(
                    src=self.node, dst=self.referee, protocol=MagicSquareGame,
                    data={'type': MagicSquareGame.Message.Type.ANSWER,
                          'answer': [self.node.qreg.units[0]['outcome'],
                                     self.node.qreg.units[1]['outcome']]
                          }
                )
                self.node.send_classical_msg(dst=self.referee, msg=answer_msg)

                # 继续准备下一轮的游戏
                if self.current_round < self.rounds:
                    self.prepare_for_game()

        def receive_quantum_msg(self) -> None:
            # 收到所有纠缠粒子后，开始魔方游戏
            if all(unit['qubit'] is not None for unit in self.node.qreg.units):
                self.play_game()  

        def play_game(self) -> None:
            # 向裁判发送消息，示意自己准备就绪，可以开始游戏
            ready_msg = MagicSquareGame.Message(
                src=self.node, dst=self.referee, protocol=MagicSquareGame,
                data={'type': MagicSquareGame.Message.Type.READY}
            )
            self.node.send_classical_msg(dst=self.referee, msg=ready_msg)
```

### 3. 玩家 2（``Player2``）

在该协议中，玩家 2 的行为包括：

1. 确认收到两份纠缠粒子后，向裁判发送 ``READY`` 信息，开始魔方游戏；
2. 收到裁判的问题 $y$（需要填入元素的列号）后，对他所持的两个粒子执行对应的操作并测量；
3. 将测量结果作为回答返回给裁判。


```python
class MagicSquareGame(Protocol):
    ...
    class Player2(SubProtocol):

        def __init__(self, super_protocol: Protocol):
            super().__init__(super_protocol)
            self.peer = None
            self.ent_source = None
            self.referee = None

        def start(self, **kwargs) -> None:
            # 另一位游戏玩家
            self.peer = kwargs['peer']  
            # 纠缠源
            self.ent_source = kwargs['ent_source']  
            # 游戏裁判
            self.referee = kwargs['referee']  

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            if msg.data['type'] == MagicSquareGame.Message.Type.QUESTION:
                # 记录需要填入元素的列号
                column = msg.data['question']  

                # 若需要填入第 0 列，对第一个粒子执行 X 基测量，对第二个粒子执行 Z 基测量
                if column == 0:
                    self.node.qreg.measure(0, basis="x")
                    self.node.qreg.measure(1, basis="z")
                    self.node.qreg.circuit.name += "c0"  # 记录玩家 2 需要填写的列号

                # 若需要填入第 1 列，对第一个粒子执行 Z 基测量，对第二个粒子执行 X 基测量
                elif column == 1:
                    self.node.qreg.measure(0, basis="z")
                    self.node.qreg.measure(1, basis="x")
                    self.node.qreg.circuit.name += "c1"  # 记录玩家 2 需要填写的列号

                # 若需要填入第 2 列，对所持粒子执行贝尔基测量
                elif column == 2:
                    self.node.qreg.bsm([0, 1])
                    self.node.qreg.circuit.name += "c2"  # 记录玩家 2 需要填写的列号

                # 将答复消息发送给裁判
                answer_msg = MagicSquareGame.Message(
                    src=self.node, dst=self.referee, protocol=MagicSquareGame,
                    data={'type': MagicSquareGame.Message.Type.ANSWER,
                          'answer': [self.node.qreg.units[0]['outcome'],
                                     self.node.qreg.units[1]['outcome']]
                          }
                )
                self.node.send_classical_msg(dst=self.referee, msg=answer_msg)

        def receive_quantum_msg(self) -> None:
            # 收到所有纠缠粒子后，开始魔方游戏
            if all(unit['qubit'] is not None for unit in self.node.qreg.units):
                self.play_game()  

        def play_game(self):
            # 向裁判发送消息，示意自己准备就绪，可以开始游戏
            ready_msg = MagicSquareGame.Message(
                src=self.node, dst=self.referee, protocol=MagicSquareGame,
                data={'type': MagicSquareGame.Message.Type.READY}
            )
            self.node.send_classical_msg(dst=self.referee, msg=ready_msg)
```

### 4. 裁判 (``Referee``)

在该协议中，裁判的行为包括：

1. 游戏开始前，等待由玩家发来的准备就绪的消息，当所有玩家准备就绪，正式开始游戏；
2. 游戏开始时，随机生成两个问题 $x, y \in \{0, 1, 2\}$ （分别指定两位玩家需要填充的行和列）并记录，随后分别将这两个问题发送给玩家 1 和玩家 2；
3. 等待两位玩家分别返回作答结果并保存；
4. 获得仿真运行的结果后，在游戏每个轮次中，根据玩家的测量结果为其填入对应的数字并判断胜负，统计其获胜概率。


```python
class MagicSquareGame(Protocol):
    ...
    class Referee(SubProtocol):

        def __init__(self, super_protocol: Protocol):
            super().__init__(super_protocol)
            # 参与游戏的玩家
            self.players = None  
            # 判断玩家是否准备就绪
            self.players_ready = [False, False]  
            # 保存每一轮游戏的提问
            self.questions = []  
            # 保存每一轮游戏中玩家 1 的回答
            self.answers_p1 = []  
            # 保存每一轮游戏中玩家 2 的回答
            self.answers_p2 = []  

        def start(self, **kwargs) -> None:
            self.players = kwargs['players']

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            if msg.data['type'] == MagicSquareGame.Message.Type.READY:
                self.players_ready[self.players.index(msg.src)] = True
                # 如果玩家都准备就绪，则开始游戏分发问题
                if all(self.players_ready):
                    self.send_questions()
                    # 为下一轮游戏做准备
                    self.players_ready = [False, False]

            elif msg.data['type'] == MagicSquareGame.Message.Type.ANSWER:
                # 收到玩家回答后进行存储
                if msg.src == self.players[0]:
                    self.answers_p1.append(msg.data['answer'])
                elif msg.src == self.players[1]:
                    self.answers_p2.append(msg.data['answer'])

        def send_questions(self) -> None:
            # 随机指定需要填充的行和列作为问题
            questions = numpy.random.choice([0, 1, 2], size=2)  
            self.questions.append(questions)

            # 将需要填充的行、列号以问题的形式分别发送给两位玩家
            for i, player in enumerate(self.players):
                question_msg = MagicSquareGame.Message(
                    src=self.node, dst=self.players[i], protocol=MagicSquareGame,
                    data={'type': MagicSquareGame.Message.Type.QUESTION, 'question': questions[i]}
                )
                self.node.send_classical_msg(dst=self.players[i], msg=question_msg)

        def estimate_statistics(self, results: List[Dict]) -> None:
            # 根据指定行和玩家 1 的测量结果补充玩家 1 的填表选择
            def p1_answer(row: int, outcome: List[int]) -> list:
                if row == 0:
                    if outcome == [0, 0]:
                        return [1, 1, 1]
                    elif outcome == [0, 1]:
                        return [-1, 1, -1]
                    elif outcome == [1, 0]:
                        return [1, -1, -1]
                    elif outcome == [1, 1]:
                        return [-1, -1, 1]

                elif row == 1 or row == 2:
                    if outcome == [0, 0]:
                        return [1, 1, 1]
                    elif outcome == [0, 1]:
                        return [1, -1, -1]
                    elif outcome == [1, 0]:
                        return [-1, 1, -1]
                    elif outcome == [1, 1]:
                        return [-1, -1, 1]

            # 根据指定列和玩家 2 的测量结果补充玩家 2 的填表选择
            def p2_answer(column: int, outcome: List[int]) -> list:
                if column == 0 or column == 2:
                    if outcome == [0, 0]:
                        return [1, 1, -1]
                    elif outcome == [0, 1]:
                        return [-1, 1, 1]
                    elif outcome == [1, 0]:
                        return [1, -1, 1]
                    elif outcome == [1, 1]:
                        return [-1, -1, -1]

                if column == 1:
                    if outcome == [0, 0]:
                        return [1, 1, -1]
                    elif outcome == [0, 1]:
                        return [1, -1, 1]
                    elif outcome == [1, 0]:
                        return [-1, 1, 1]
                    elif outcome == [1, 1]:
                        return [-1, -1, -1]

            # 判断玩家是否获胜
            def is_winning(row: int, column: int, answer_p1: List[int], answer_p2: List[int]) -> bool:
                # 若玩家 1 所填行元素乘积为 +1，玩家 2 所填列元素乘积为 -1，且行列相交处元素相同，则玩家取胜
                if numpy.prod(answer_p1) == 1 and numpy.prod(answer_p2) == -1 and answer_p1[column] == answer_p2[row]:
                    return True
                # 否则，玩家失败
                else:
                    return False

            num_wins = 0  # 用于统计玩家的获胜轮数

            for i, result in enumerate(results):
                cir_name = result['circuit_name']
                counts = result['counts']

                if "r0" in cir_name:  # 玩家 1：填充第 0 行
                    row = 0
                elif "r1" in cir_name:  # 玩家 1：填充第 1 行
                    row = 1
                elif "r2" in cir_name:  # 玩家 1：填充第 2 行
                    row = 2

                if "c0" in cir_name:  # 玩家 2：填充第 0 列
                    column = 0
                elif "c1" in cir_name:  # 玩家 2：填充第 1 列
                    column = 1
                elif "c2" in cir_name:  # 玩家 2：填充第 2 列
                    column = 2

                for count in counts:
                    answer_p1, answer_p2 = self.answers_p1[0], self.answers_p2[0]
                    # 玩家 1 的测量结果
                    outcome_p1 = [int(count[answer_p1[0]]), int(count[answer_p1[1]])]  
                    # 玩家 2 的测量结果
                    outcome_p2 = [int(count[answer_p2[0]]), int(count[answer_p2[1]])]  

                    if is_winning(row, column, p1_answer(row, outcome_p1), p2_answer(column, outcome_p2)):
                        num_wins += counts[count]  # 统计所有获胜的游戏轮数
            # 统计获胜概率
            winning_prob = num_wins / sum(result['shots'] for result in results)  
            print(f"\n{'-' * 60}\nThe winning probability of the magic square game is {winning_prob:.4f}.\n{'-' * 60}")
```

## 代码示例

接下来我们使用 QNET 对量子最优策略下的魔方游戏进行仿真。

首先，创建一个仿真环境 ``QuantumEnv``。


```python
from qcompute_qnet.models.qpu.env import QuantumEnv

# 创建一个仿真环境
env = QuantumEnv("Magic square game", default=True)
```

随后，分别创建该游戏中四种角色所对应的量子节点，设定这些节点的协议栈中预装的协议为 ``MagicSquareGame``，并配置这些节点之间必要的通信链路。


```python
from qcompute_qnet.models.qpu.node import QuantumNode
from qcompute_qnet.models.qpu.protocol import MagicSquareGame
from qcompute_qnet.topology.link import Link

# 创建装有量子寄存器的量子节点并指定其中预装的协议类型
alice = QuantumNode("Alice", qreg_size=2, protocol=MagicSquareGame)
bob = QuantumNode("Bob", qreg_size=2, protocol=MagicSquareGame)
source = QuantumNode("Source", qreg_size=4, protocol=MagicSquareGame)
referee = QuantumNode("Referee", qreg_size=0, protocol=MagicSquareGame)

# 创建节点间的通信链路
link_as = Link("link_as", ends=(alice, source), distance=1e3)
link_bs = Link("link_bs", ends=(bob, source), distance=1e3)
link_ar = Link("link_ar", ends=(alice, referee), distance=1e3)
link_br = Link("link_br", ends=(bob, referee), distance=1e3)
```

然后，创建一个量子网络，并将配置好的节点和通信链路装入量子网络中。


```python
from qcompute_qnet.topology.network import Network

# 创建一个量子网络并将各节点和各链路装入量子网络中
network = Network("Magic square game network")
network.install([alice, bob, referee, source, link_as, link_bs, link_ar, link_br])
```

现在我们完成了对仿真环境和量子网络的搭建，接下来可以通过节点的 ``start`` 方法，启动节点协议栈中装载的 ``MagicSquareGame`` 协议。在这里，我们设置游戏总轮数为 1024 次。


```python
# 游戏总轮数
game_rounds = 1024

# 启动魔方游戏协议
alice.start(role="Player1", peer=bob, ent_source=source, referee=referee, rounds=game_rounds)
bob.start(role="Player2", peer=alice, ent_source=source, referee=referee)
source.start(role="Source")
referee.start(role="Referee", players=[alice, bob])
```

最后，对仿真环境进行初始化并运行，并获得输出结果。

通过 ``Referee`` 的 ``estimate_statistics`` 方法对电路运行结果进行分析，我们可以依次判断游戏中每轮次玩家的胜负情况，并统计玩家的获胜概率。


```python
from qcompute_qnet.quantum.backends import Backend

# 初始化仿真环境并运行仿真
env.init()
results = env.run(backend=Backend.QCompute.LocalBaiduSim2, summary=False)

# 对玩家的获胜概率进行统计
referee.protocol.estimate_statistics(results)
```

仿真运行后，我们将在终端上看到以下 9 种不同量子电路的打印结果，分别对应着不同的游戏操作。

其中，电路中红色的部分代表纠缠源的操作，蓝色代表玩家 1 的操作，绿色代表玩家 2 的操作。

![图 6：量子魔方游戏协议编译成的量子电路图](./figures/magic_square_game-circuits.png "图 6：量子魔方游戏协议编译成的量子电路图")

统计所得的获胜概率如下所示：

```
The winning probability of the magic square game is 1.0000.
```

我们可以发现，通过采用量子最优策略，无论裁判如何随机发问，玩家都能以 $100\%$ 的几率获得魔方游戏的胜利！通过这个有趣的例子，我们也验证了，在某些问题上，使用量子策略确实有着经典策略所无法比拟的优越性。

---

## 参考文献

[1] Cirel'son, Boris S. "Quantum generalizations of Bell's inequality." [Letters in Mathematical Physics 4.2 (1980): 93-100.](https://link.springer.com/article/10.1007/BF00417500)

[2] Mermin, N. David. "Simple unified form for the major no-hidden-variables theorems." [Physical Review Letters 65.27 (1990): 3373.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.65.3373)

[3] Mermin, N. David. "Hidden variables and the two theorems of John Bell." [Reviews of Modern Physics 65.3 (1993): 803.](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.65.803)
