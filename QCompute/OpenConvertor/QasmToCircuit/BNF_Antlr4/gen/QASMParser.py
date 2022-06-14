# Generated from C:/Project/Quantum/QCompute/QCompute/OpenConvertor/QasmToCircuit/BNF_Antlr4\QASM.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,37,323,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        1,0,1,0,1,0,5,0,58,8,0,10,0,12,0,61,9,0,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,93,8,1,1,2,1,2,1,2,1,2,1,3,1,
        3,1,3,1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,
        4,3,4,117,8,4,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,
        5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,3,5,139,8,5,1,6,1,6,1,6,4,6,144,8,
        6,11,6,12,6,145,1,7,1,7,1,7,3,7,151,8,7,1,8,1,8,1,8,1,8,1,8,1,8,
        1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,3,8,170,8,8,1,9,1,9,
        3,9,174,8,9,1,10,1,10,1,11,1,11,1,12,1,12,1,12,1,12,1,12,1,12,1,
        13,1,13,1,13,1,13,1,14,1,14,3,14,192,8,14,1,15,1,15,5,15,196,8,15,
        10,15,12,15,199,9,15,1,15,1,15,1,16,1,16,1,16,1,16,1,16,5,16,208,
        8,16,10,16,12,16,211,9,16,1,16,1,16,1,16,1,16,1,16,1,16,5,16,219,
        8,16,10,16,12,16,222,9,16,1,16,1,16,1,16,1,16,1,16,1,16,5,16,230,
        8,16,10,16,12,16,233,9,16,1,16,1,16,5,16,237,8,16,10,16,12,16,240,
        9,16,1,16,1,16,1,16,1,16,3,16,246,8,16,1,17,1,17,1,17,1,17,1,17,
        3,17,253,8,17,1,18,1,18,1,18,5,18,258,8,18,10,18,12,18,261,9,18,
        1,18,1,18,1,19,1,19,1,19,1,19,1,19,1,19,1,19,1,19,1,19,1,19,1,19,
        1,19,1,19,1,19,1,19,1,19,3,19,281,8,19,1,19,1,19,1,19,1,19,1,19,
        1,19,1,19,1,19,1,19,1,19,1,19,1,19,1,19,1,19,1,19,5,19,298,8,19,
        10,19,12,19,301,9,19,1,20,1,20,1,20,1,20,1,20,1,20,3,20,309,8,20,
        1,21,1,21,1,22,1,22,1,23,1,23,1,24,1,24,1,25,1,25,1,26,1,26,1,26,
        0,1,38,27,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,
        40,42,44,46,48,50,52,0,0,340,0,54,1,0,0,0,2,92,1,0,0,0,4,94,1,0,
        0,0,6,98,1,0,0,0,8,116,1,0,0,0,10,138,1,0,0,0,12,143,1,0,0,0,14,
        150,1,0,0,0,16,169,1,0,0,0,18,173,1,0,0,0,20,175,1,0,0,0,22,177,
        1,0,0,0,24,179,1,0,0,0,26,185,1,0,0,0,28,191,1,0,0,0,30,197,1,0,
        0,0,32,245,1,0,0,0,34,252,1,0,0,0,36,259,1,0,0,0,38,280,1,0,0,0,
        40,308,1,0,0,0,42,310,1,0,0,0,44,312,1,0,0,0,46,314,1,0,0,0,48,316,
        1,0,0,0,50,318,1,0,0,0,52,320,1,0,0,0,54,55,3,4,2,0,55,59,3,6,3,
        0,56,58,3,2,1,0,57,56,1,0,0,0,58,61,1,0,0,0,59,57,1,0,0,0,59,60,
        1,0,0,0,60,1,1,0,0,0,61,59,1,0,0,0,62,93,3,8,4,0,63,64,3,10,5,0,
        64,65,3,12,6,0,65,66,5,1,0,0,66,93,1,0,0,0,67,68,3,10,5,0,68,69,
        5,1,0,0,69,93,1,0,0,0,70,71,5,2,0,0,71,72,5,31,0,0,72,73,3,30,15,
        0,73,74,5,3,0,0,74,93,1,0,0,0,75,76,5,2,0,0,76,77,5,31,0,0,77,78,
        5,4,0,0,78,79,5,5,0,0,79,80,3,30,15,0,80,81,5,3,0,0,81,93,1,0,0,
        0,82,83,5,2,0,0,83,84,5,31,0,0,84,85,5,4,0,0,85,86,3,30,15,0,86,
        87,5,5,0,0,87,88,3,30,15,0,88,89,5,3,0,0,89,93,1,0,0,0,90,93,3,14,
        7,0,91,93,3,18,9,0,92,62,1,0,0,0,92,63,1,0,0,0,92,67,1,0,0,0,92,
        70,1,0,0,0,92,75,1,0,0,0,92,82,1,0,0,0,92,90,1,0,0,0,92,91,1,0,0,
        0,93,3,1,0,0,0,94,95,5,6,0,0,95,96,5,32,0,0,96,97,5,3,0,0,97,5,1,
        0,0,0,98,99,5,7,0,0,99,100,5,8,0,0,100,101,5,37,0,0,101,102,5,8,
        0,0,102,103,5,3,0,0,103,7,1,0,0,0,104,105,5,9,0,0,105,106,5,31,0,
        0,106,107,5,10,0,0,107,108,5,33,0,0,108,109,5,11,0,0,109,117,5,3,
        0,0,110,111,5,12,0,0,111,112,5,31,0,0,112,113,5,10,0,0,113,114,5,
        33,0,0,114,115,5,11,0,0,115,117,5,3,0,0,116,104,1,0,0,0,116,110,
        1,0,0,0,117,9,1,0,0,0,118,119,5,13,0,0,119,120,5,31,0,0,120,121,
        3,30,15,0,121,122,5,14,0,0,122,139,1,0,0,0,123,124,5,13,0,0,124,
        125,5,31,0,0,125,126,5,4,0,0,126,127,5,5,0,0,127,128,3,30,15,0,128,
        129,5,14,0,0,129,139,1,0,0,0,130,131,5,13,0,0,131,132,5,31,0,0,132,
        133,5,4,0,0,133,134,3,30,15,0,134,135,5,5,0,0,135,136,3,30,15,0,
        136,137,5,14,0,0,137,139,1,0,0,0,138,118,1,0,0,0,138,123,1,0,0,0,
        138,130,1,0,0,0,139,11,1,0,0,0,140,144,3,16,8,0,141,144,3,26,13,
        0,142,144,3,18,9,0,143,140,1,0,0,0,143,141,1,0,0,0,143,142,1,0,0,
        0,144,145,1,0,0,0,145,143,1,0,0,0,145,146,1,0,0,0,146,13,1,0,0,0,
        147,151,3,16,8,0,148,151,3,24,12,0,149,151,3,26,13,0,150,147,1,0,
        0,0,150,148,1,0,0,0,150,149,1,0,0,0,151,15,1,0,0,0,152,153,5,31,
        0,0,153,154,3,28,14,0,154,155,5,3,0,0,155,170,1,0,0,0,156,157,5,
        31,0,0,157,158,5,4,0,0,158,159,5,5,0,0,159,160,3,28,14,0,160,161,
        5,3,0,0,161,170,1,0,0,0,162,163,5,31,0,0,163,164,5,4,0,0,164,165,
        3,36,18,0,165,166,5,5,0,0,166,167,3,28,14,0,167,168,5,3,0,0,168,
        170,1,0,0,0,169,152,1,0,0,0,169,156,1,0,0,0,169,162,1,0,0,0,170,
        17,1,0,0,0,171,174,3,20,10,0,172,174,3,22,11,0,173,171,1,0,0,0,173,
        172,1,0,0,0,174,19,1,0,0,0,175,176,5,36,0,0,176,21,1,0,0,0,177,178,
        5,35,0,0,178,23,1,0,0,0,179,180,5,15,0,0,180,181,3,34,17,0,181,182,
        5,16,0,0,182,183,3,34,17,0,183,184,5,3,0,0,184,25,1,0,0,0,185,186,
        5,17,0,0,186,187,3,28,14,0,187,188,5,3,0,0,188,27,1,0,0,0,189,192,
        3,30,15,0,190,192,3,32,16,0,191,189,1,0,0,0,191,190,1,0,0,0,192,
        29,1,0,0,0,193,194,5,31,0,0,194,196,5,18,0,0,195,193,1,0,0,0,196,
        199,1,0,0,0,197,195,1,0,0,0,197,198,1,0,0,0,198,200,1,0,0,0,199,
        197,1,0,0,0,200,201,5,31,0,0,201,31,1,0,0,0,202,203,5,31,0,0,203,
        204,5,10,0,0,204,205,5,33,0,0,205,206,5,11,0,0,206,208,5,18,0,0,
        207,202,1,0,0,0,208,211,1,0,0,0,209,207,1,0,0,0,209,210,1,0,0,0,
        210,212,1,0,0,0,211,209,1,0,0,0,212,246,5,31,0,0,213,214,5,31,0,
        0,214,215,5,10,0,0,215,216,5,33,0,0,216,217,5,11,0,0,217,219,5,18,
        0,0,218,213,1,0,0,0,219,222,1,0,0,0,220,218,1,0,0,0,220,221,1,0,
        0,0,221,223,1,0,0,0,222,220,1,0,0,0,223,224,5,31,0,0,224,225,5,10,
        0,0,225,226,5,33,0,0,226,246,5,11,0,0,227,228,5,31,0,0,228,230,5,
        18,0,0,229,227,1,0,0,0,230,233,1,0,0,0,231,229,1,0,0,0,231,232,1,
        0,0,0,232,234,1,0,0,0,233,231,1,0,0,0,234,235,5,31,0,0,235,237,5,
        18,0,0,236,231,1,0,0,0,237,240,1,0,0,0,238,236,1,0,0,0,238,239,1,
        0,0,0,239,241,1,0,0,0,240,238,1,0,0,0,241,242,5,31,0,0,242,243,5,
        10,0,0,243,244,5,33,0,0,244,246,5,11,0,0,245,209,1,0,0,0,245,220,
        1,0,0,0,245,238,1,0,0,0,246,33,1,0,0,0,247,253,5,31,0,0,248,249,
        5,31,0,0,249,250,5,10,0,0,250,251,5,33,0,0,251,253,5,11,0,0,252,
        247,1,0,0,0,252,248,1,0,0,0,253,35,1,0,0,0,254,255,3,38,19,0,255,
        256,5,18,0,0,256,258,1,0,0,0,257,254,1,0,0,0,258,261,1,0,0,0,259,
        257,1,0,0,0,259,260,1,0,0,0,260,262,1,0,0,0,261,259,1,0,0,0,262,
        263,3,38,19,0,263,37,1,0,0,0,264,265,6,19,-1,0,265,281,5,32,0,0,
        266,281,5,33,0,0,267,281,5,19,0,0,268,281,5,31,0,0,269,270,5,21,
        0,0,270,281,3,38,19,4,271,272,5,4,0,0,272,273,3,38,19,0,273,274,
        5,5,0,0,274,281,1,0,0,0,275,276,3,40,20,0,276,277,5,4,0,0,277,278,
        3,38,19,0,278,279,5,5,0,0,279,281,1,0,0,0,280,264,1,0,0,0,280,266,
        1,0,0,0,280,267,1,0,0,0,280,268,1,0,0,0,280,269,1,0,0,0,280,271,
        1,0,0,0,280,275,1,0,0,0,281,299,1,0,0,0,282,283,10,8,0,0,283,284,
        5,20,0,0,284,298,3,38,19,9,285,286,10,7,0,0,286,287,5,21,0,0,287,
        298,3,38,19,8,288,289,10,6,0,0,289,290,5,22,0,0,290,298,3,38,19,
        7,291,292,10,5,0,0,292,293,5,23,0,0,293,298,3,38,19,6,294,295,10,
        3,0,0,295,296,5,24,0,0,296,298,3,38,19,4,297,282,1,0,0,0,297,285,
        1,0,0,0,297,288,1,0,0,0,297,291,1,0,0,0,297,294,1,0,0,0,298,301,
        1,0,0,0,299,297,1,0,0,0,299,300,1,0,0,0,300,39,1,0,0,0,301,299,1,
        0,0,0,302,309,3,42,21,0,303,309,3,44,22,0,304,309,3,46,23,0,305,
        309,3,48,24,0,306,309,3,50,25,0,307,309,3,52,26,0,308,302,1,0,0,
        0,308,303,1,0,0,0,308,304,1,0,0,0,308,305,1,0,0,0,308,306,1,0,0,
        0,308,307,1,0,0,0,309,41,1,0,0,0,310,311,5,25,0,0,311,43,1,0,0,0,
        312,313,5,26,0,0,313,45,1,0,0,0,314,315,5,27,0,0,315,47,1,0,0,0,
        316,317,5,28,0,0,317,49,1,0,0,0,318,319,5,29,0,0,319,51,1,0,0,0,
        320,321,5,30,0,0,321,53,1,0,0,0,22,59,92,116,138,143,145,150,169,
        173,191,197,209,220,231,238,245,252,259,280,297,299,308
    ]

class QASMParser ( Parser ):

    grammarFileName = "QASM.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'}'", "'opaque'", "';'", "'('", "')'", 
                     "'OPENQASM'", "'include'", "'\"'", "'qreg'", "'['", 
                     "']'", "'creg'", "'gate'", "'{'", "'measure'", "'->'", 
                     "'barrier'", "','", "'pi'", "'+'", "'-'", "'*'", "'/'", 
                     "'^'", "'sin'", "'cos'", "'tan'", "'exp'", "'ln'", 
                     "'sqrt'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "ID", "REAL", 
                      "INT", "WS", "COMMENT", "LINE_COMMENT", "FILE_NAME" ]

    RULE_mainprog = 0
    RULE_statement = 1
    RULE_version = 2
    RULE_include = 3
    RULE_decl = 4
    RULE_gatedecl = 5
    RULE_goplist = 6
    RULE_qop = 7
    RULE_uop = 8
    RULE_comments = 9
    RULE_lineComment = 10
    RULE_multipleComments = 11
    RULE_measureOp = 12
    RULE_barrierOp = 13
    RULE_anylist = 14
    RULE_idlist = 15
    RULE_mixedlist = 16
    RULE_argument = 17
    RULE_explist = 18
    RULE_exp = 19
    RULE_unaryop = 20
    RULE_sin = 21
    RULE_cos = 22
    RULE_tan = 23
    RULE_expx = 24
    RULE_ln = 25
    RULE_sqrt = 26

    ruleNames =  [ "mainprog", "statement", "version", "include", "decl", 
                   "gatedecl", "goplist", "qop", "uop", "comments", "lineComment", 
                   "multipleComments", "measureOp", "barrierOp", "anylist", 
                   "idlist", "mixedlist", "argument", "explist", "exp", 
                   "unaryop", "sin", "cos", "tan", "expx", "ln", "sqrt" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    T__14=15
    T__15=16
    T__16=17
    T__17=18
    T__18=19
    T__19=20
    T__20=21
    T__21=22
    T__22=23
    T__23=24
    T__24=25
    T__25=26
    T__26=27
    T__27=28
    T__28=29
    T__29=30
    ID=31
    REAL=32
    INT=33
    WS=34
    COMMENT=35
    LINE_COMMENT=36
    FILE_NAME=37

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class MainprogContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def version(self):
            return self.getTypedRuleContext(QASMParser.VersionContext,0)


        def include(self):
            return self.getTypedRuleContext(QASMParser.IncludeContext,0)


        def statement(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(QASMParser.StatementContext)
            else:
                return self.getTypedRuleContext(QASMParser.StatementContext,i)


        def getRuleIndex(self):
            return QASMParser.RULE_mainprog

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMainprog" ):
                listener.enterMainprog(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMainprog" ):
                listener.exitMainprog(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMainprog" ):
                return visitor.visitMainprog(self)
            else:
                return visitor.visitChildren(self)




    def mainprog(self):

        localctx = QASMParser.MainprogContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_mainprog)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 54
            self.version()
            self.state = 55
            self.include()
            self.state = 59
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << QASMParser.T__1) | (1 << QASMParser.T__8) | (1 << QASMParser.T__11) | (1 << QASMParser.T__12) | (1 << QASMParser.T__14) | (1 << QASMParser.T__16) | (1 << QASMParser.ID) | (1 << QASMParser.COMMENT) | (1 << QASMParser.LINE_COMMENT))) != 0):
                self.state = 56
                self.statement()
                self.state = 61
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class StatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def decl(self):
            return self.getTypedRuleContext(QASMParser.DeclContext,0)


        def gatedecl(self):
            return self.getTypedRuleContext(QASMParser.GatedeclContext,0)


        def goplist(self):
            return self.getTypedRuleContext(QASMParser.GoplistContext,0)


        def ID(self):
            return self.getToken(QASMParser.ID, 0)

        def idlist(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(QASMParser.IdlistContext)
            else:
                return self.getTypedRuleContext(QASMParser.IdlistContext,i)


        def qop(self):
            return self.getTypedRuleContext(QASMParser.QopContext,0)


        def comments(self):
            return self.getTypedRuleContext(QASMParser.CommentsContext,0)


        def getRuleIndex(self):
            return QASMParser.RULE_statement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStatement" ):
                listener.enterStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStatement" ):
                listener.exitStatement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStatement" ):
                return visitor.visitStatement(self)
            else:
                return visitor.visitChildren(self)




    def statement(self):

        localctx = QASMParser.StatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_statement)
        try:
            self.state = 92
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 62
                self.decl()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 63
                self.gatedecl()
                self.state = 64
                self.goplist()
                self.state = 65
                self.match(QASMParser.T__0)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 67
                self.gatedecl()
                self.state = 68
                self.match(QASMParser.T__0)
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 70
                self.match(QASMParser.T__1)
                self.state = 71
                self.match(QASMParser.ID)
                self.state = 72
                self.idlist()
                self.state = 73
                self.match(QASMParser.T__2)
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 75
                self.match(QASMParser.T__1)
                self.state = 76
                self.match(QASMParser.ID)
                self.state = 77
                self.match(QASMParser.T__3)
                self.state = 78
                self.match(QASMParser.T__4)
                self.state = 79
                self.idlist()
                self.state = 80
                self.match(QASMParser.T__2)
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 82
                self.match(QASMParser.T__1)
                self.state = 83
                self.match(QASMParser.ID)
                self.state = 84
                self.match(QASMParser.T__3)
                self.state = 85
                self.idlist()
                self.state = 86
                self.match(QASMParser.T__4)
                self.state = 87
                self.idlist()
                self.state = 88
                self.match(QASMParser.T__2)
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 90
                self.qop()
                pass

            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 91
                self.comments()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class VersionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def REAL(self):
            return self.getToken(QASMParser.REAL, 0)

        def getRuleIndex(self):
            return QASMParser.RULE_version

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVersion" ):
                listener.enterVersion(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVersion" ):
                listener.exitVersion(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVersion" ):
                return visitor.visitVersion(self)
            else:
                return visitor.visitChildren(self)




    def version(self):

        localctx = QASMParser.VersionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_version)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 94
            self.match(QASMParser.T__5)
            self.state = 95
            self.match(QASMParser.REAL)
            self.state = 96
            self.match(QASMParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IncludeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def FILE_NAME(self):
            return self.getToken(QASMParser.FILE_NAME, 0)

        def getRuleIndex(self):
            return QASMParser.RULE_include

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInclude" ):
                listener.enterInclude(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInclude" ):
                listener.exitInclude(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitInclude" ):
                return visitor.visitInclude(self)
            else:
                return visitor.visitChildren(self)




    def include(self):

        localctx = QASMParser.IncludeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_include)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 98
            self.match(QASMParser.T__6)
            self.state = 99
            self.match(QASMParser.T__7)
            self.state = 100
            self.match(QASMParser.FILE_NAME)
            self.state = 101
            self.match(QASMParser.T__7)
            self.state = 102
            self.match(QASMParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DeclContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return QASMParser.RULE_decl

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class QRegContext(DeclContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a QASMParser.DeclContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(QASMParser.ID, 0)
        def INT(self):
            return self.getToken(QASMParser.INT, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterQReg" ):
                listener.enterQReg(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitQReg" ):
                listener.exitQReg(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitQReg" ):
                return visitor.visitQReg(self)
            else:
                return visitor.visitChildren(self)


    class CRegContext(DeclContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a QASMParser.DeclContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(QASMParser.ID, 0)
        def INT(self):
            return self.getToken(QASMParser.INT, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCReg" ):
                listener.enterCReg(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCReg" ):
                listener.exitCReg(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCReg" ):
                return visitor.visitCReg(self)
            else:
                return visitor.visitChildren(self)



    def decl(self):

        localctx = QASMParser.DeclContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_decl)
        try:
            self.state = 116
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [QASMParser.T__8]:
                localctx = QASMParser.QRegContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 104
                self.match(QASMParser.T__8)
                self.state = 105
                self.match(QASMParser.ID)
                self.state = 106
                self.match(QASMParser.T__9)
                self.state = 107
                self.match(QASMParser.INT)
                self.state = 108
                self.match(QASMParser.T__10)
                self.state = 109
                self.match(QASMParser.T__2)
                pass
            elif token in [QASMParser.T__11]:
                localctx = QASMParser.CRegContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 110
                self.match(QASMParser.T__11)
                self.state = 111
                self.match(QASMParser.ID)
                self.state = 112
                self.match(QASMParser.T__9)
                self.state = 113
                self.match(QASMParser.INT)
                self.state = 114
                self.match(QASMParser.T__10)
                self.state = 115
                self.match(QASMParser.T__2)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GatedeclContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(QASMParser.ID, 0)

        def idlist(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(QASMParser.IdlistContext)
            else:
                return self.getTypedRuleContext(QASMParser.IdlistContext,i)


        def getRuleIndex(self):
            return QASMParser.RULE_gatedecl

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterGatedecl" ):
                listener.enterGatedecl(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitGatedecl" ):
                listener.exitGatedecl(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGatedecl" ):
                return visitor.visitGatedecl(self)
            else:
                return visitor.visitChildren(self)




    def gatedecl(self):

        localctx = QASMParser.GatedeclContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_gatedecl)
        try:
            self.state = 138
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 118
                self.match(QASMParser.T__12)
                self.state = 119
                self.match(QASMParser.ID)
                self.state = 120
                self.idlist()
                self.state = 121
                self.match(QASMParser.T__13)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 123
                self.match(QASMParser.T__12)
                self.state = 124
                self.match(QASMParser.ID)
                self.state = 125
                self.match(QASMParser.T__3)
                self.state = 126
                self.match(QASMParser.T__4)
                self.state = 127
                self.idlist()
                self.state = 128
                self.match(QASMParser.T__13)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 130
                self.match(QASMParser.T__12)
                self.state = 131
                self.match(QASMParser.ID)
                self.state = 132
                self.match(QASMParser.T__3)
                self.state = 133
                self.idlist()
                self.state = 134
                self.match(QASMParser.T__4)
                self.state = 135
                self.idlist()
                self.state = 136
                self.match(QASMParser.T__13)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GoplistContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def uop(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(QASMParser.UopContext)
            else:
                return self.getTypedRuleContext(QASMParser.UopContext,i)


        def barrierOp(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(QASMParser.BarrierOpContext)
            else:
                return self.getTypedRuleContext(QASMParser.BarrierOpContext,i)


        def comments(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(QASMParser.CommentsContext)
            else:
                return self.getTypedRuleContext(QASMParser.CommentsContext,i)


        def getRuleIndex(self):
            return QASMParser.RULE_goplist

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterGoplist" ):
                listener.enterGoplist(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitGoplist" ):
                listener.exitGoplist(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitGoplist" ):
                return visitor.visitGoplist(self)
            else:
                return visitor.visitChildren(self)




    def goplist(self):

        localctx = QASMParser.GoplistContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_goplist)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 143 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 143
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [QASMParser.ID]:
                    self.state = 140
                    self.uop()
                    pass
                elif token in [QASMParser.T__16]:
                    self.state = 141
                    self.barrierOp()
                    pass
                elif token in [QASMParser.COMMENT, QASMParser.LINE_COMMENT]:
                    self.state = 142
                    self.comments()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 145 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not ((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << QASMParser.T__16) | (1 << QASMParser.ID) | (1 << QASMParser.COMMENT) | (1 << QASMParser.LINE_COMMENT))) != 0)):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class QopContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def uop(self):
            return self.getTypedRuleContext(QASMParser.UopContext,0)


        def measureOp(self):
            return self.getTypedRuleContext(QASMParser.MeasureOpContext,0)


        def barrierOp(self):
            return self.getTypedRuleContext(QASMParser.BarrierOpContext,0)


        def getRuleIndex(self):
            return QASMParser.RULE_qop

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterQop" ):
                listener.enterQop(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitQop" ):
                listener.exitQop(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitQop" ):
                return visitor.visitQop(self)
            else:
                return visitor.visitChildren(self)




    def qop(self):

        localctx = QASMParser.QopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_qop)
        try:
            self.state = 150
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [QASMParser.ID]:
                self.enterOuterAlt(localctx, 1)
                self.state = 147
                self.uop()
                pass
            elif token in [QASMParser.T__14]:
                self.enterOuterAlt(localctx, 2)
                self.state = 148
                self.measureOp()
                pass
            elif token in [QASMParser.T__16]:
                self.enterOuterAlt(localctx, 3)
                self.state = 149
                self.barrierOp()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class UopContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(QASMParser.ID, 0)

        def anylist(self):
            return self.getTypedRuleContext(QASMParser.AnylistContext,0)


        def explist(self):
            return self.getTypedRuleContext(QASMParser.ExplistContext,0)


        def getRuleIndex(self):
            return QASMParser.RULE_uop

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterUop" ):
                listener.enterUop(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitUop" ):
                listener.exitUop(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitUop" ):
                return visitor.visitUop(self)
            else:
                return visitor.visitChildren(self)




    def uop(self):

        localctx = QASMParser.UopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_uop)
        try:
            self.state = 169
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 152
                self.match(QASMParser.ID)
                self.state = 153
                self.anylist()
                self.state = 154
                self.match(QASMParser.T__2)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 156
                self.match(QASMParser.ID)
                self.state = 157
                self.match(QASMParser.T__3)
                self.state = 158
                self.match(QASMParser.T__4)
                self.state = 159
                self.anylist()
                self.state = 160
                self.match(QASMParser.T__2)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 162
                self.match(QASMParser.ID)
                self.state = 163
                self.match(QASMParser.T__3)
                self.state = 164
                self.explist()
                self.state = 165
                self.match(QASMParser.T__4)
                self.state = 166
                self.anylist()
                self.state = 167
                self.match(QASMParser.T__2)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CommentsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def lineComment(self):
            return self.getTypedRuleContext(QASMParser.LineCommentContext,0)


        def multipleComments(self):
            return self.getTypedRuleContext(QASMParser.MultipleCommentsContext,0)


        def getRuleIndex(self):
            return QASMParser.RULE_comments

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterComments" ):
                listener.enterComments(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitComments" ):
                listener.exitComments(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitComments" ):
                return visitor.visitComments(self)
            else:
                return visitor.visitChildren(self)




    def comments(self):

        localctx = QASMParser.CommentsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_comments)
        try:
            self.state = 173
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [QASMParser.LINE_COMMENT]:
                self.enterOuterAlt(localctx, 1)
                self.state = 171
                self.lineComment()
                pass
            elif token in [QASMParser.COMMENT]:
                self.enterOuterAlt(localctx, 2)
                self.state = 172
                self.multipleComments()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LineCommentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LINE_COMMENT(self):
            return self.getToken(QASMParser.LINE_COMMENT, 0)

        def getRuleIndex(self):
            return QASMParser.RULE_lineComment

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLineComment" ):
                listener.enterLineComment(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLineComment" ):
                listener.exitLineComment(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLineComment" ):
                return visitor.visitLineComment(self)
            else:
                return visitor.visitChildren(self)




    def lineComment(self):

        localctx = QASMParser.LineCommentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_lineComment)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 175
            self.match(QASMParser.LINE_COMMENT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MultipleCommentsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def COMMENT(self):
            return self.getToken(QASMParser.COMMENT, 0)

        def getRuleIndex(self):
            return QASMParser.RULE_multipleComments

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMultipleComments" ):
                listener.enterMultipleComments(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMultipleComments" ):
                listener.exitMultipleComments(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMultipleComments" ):
                return visitor.visitMultipleComments(self)
            else:
                return visitor.visitChildren(self)




    def multipleComments(self):

        localctx = QASMParser.MultipleCommentsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_multipleComments)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 177
            self.match(QASMParser.COMMENT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MeasureOpContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def argument(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(QASMParser.ArgumentContext)
            else:
                return self.getTypedRuleContext(QASMParser.ArgumentContext,i)


        def getRuleIndex(self):
            return QASMParser.RULE_measureOp

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMeasureOp" ):
                listener.enterMeasureOp(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMeasureOp" ):
                listener.exitMeasureOp(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMeasureOp" ):
                return visitor.visitMeasureOp(self)
            else:
                return visitor.visitChildren(self)




    def measureOp(self):

        localctx = QASMParser.MeasureOpContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_measureOp)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 179
            self.match(QASMParser.T__14)
            self.state = 180
            self.argument()
            self.state = 181
            self.match(QASMParser.T__15)
            self.state = 182
            self.argument()
            self.state = 183
            self.match(QASMParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BarrierOpContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def anylist(self):
            return self.getTypedRuleContext(QASMParser.AnylistContext,0)


        def getRuleIndex(self):
            return QASMParser.RULE_barrierOp

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBarrierOp" ):
                listener.enterBarrierOp(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBarrierOp" ):
                listener.exitBarrierOp(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBarrierOp" ):
                return visitor.visitBarrierOp(self)
            else:
                return visitor.visitChildren(self)




    def barrierOp(self):

        localctx = QASMParser.BarrierOpContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_barrierOp)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 185
            self.match(QASMParser.T__16)
            self.state = 186
            self.anylist()
            self.state = 187
            self.match(QASMParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AnylistContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def idlist(self):
            return self.getTypedRuleContext(QASMParser.IdlistContext,0)


        def mixedlist(self):
            return self.getTypedRuleContext(QASMParser.MixedlistContext,0)


        def getRuleIndex(self):
            return QASMParser.RULE_anylist

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAnylist" ):
                listener.enterAnylist(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAnylist" ):
                listener.exitAnylist(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAnylist" ):
                return visitor.visitAnylist(self)
            else:
                return visitor.visitChildren(self)




    def anylist(self):

        localctx = QASMParser.AnylistContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_anylist)
        try:
            self.state = 191
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,9,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 189
                self.idlist()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 190
                self.mixedlist()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IdlistContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(QASMParser.ID)
            else:
                return self.getToken(QASMParser.ID, i)

        def getRuleIndex(self):
            return QASMParser.RULE_idlist

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIdlist" ):
                listener.enterIdlist(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIdlist" ):
                listener.exitIdlist(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIdlist" ):
                return visitor.visitIdlist(self)
            else:
                return visitor.visitChildren(self)




    def idlist(self):

        localctx = QASMParser.IdlistContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_idlist)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 197
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,10,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 193
                    self.match(QASMParser.ID)
                    self.state = 194
                    self.match(QASMParser.T__17) 
                self.state = 199
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,10,self._ctx)

            self.state = 200
            self.match(QASMParser.ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MixedlistContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(QASMParser.ID)
            else:
                return self.getToken(QASMParser.ID, i)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(QASMParser.INT)
            else:
                return self.getToken(QASMParser.INT, i)

        def getRuleIndex(self):
            return QASMParser.RULE_mixedlist

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMixedlist" ):
                listener.enterMixedlist(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMixedlist" ):
                listener.exitMixedlist(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMixedlist" ):
                return visitor.visitMixedlist(self)
            else:
                return visitor.visitChildren(self)




    def mixedlist(self):

        localctx = QASMParser.MixedlistContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_mixedlist)
        try:
            self.state = 245
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,15,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 209
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,11,self._ctx)
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt==1:
                        self.state = 202
                        self.match(QASMParser.ID)
                        self.state = 203
                        self.match(QASMParser.T__9)
                        self.state = 204
                        self.match(QASMParser.INT)
                        self.state = 205
                        self.match(QASMParser.T__10)
                        self.state = 206
                        self.match(QASMParser.T__17) 
                    self.state = 211
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,11,self._ctx)

                self.state = 212
                self.match(QASMParser.ID)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 220
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,12,self._ctx)
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt==1:
                        self.state = 213
                        self.match(QASMParser.ID)
                        self.state = 214
                        self.match(QASMParser.T__9)
                        self.state = 215
                        self.match(QASMParser.INT)
                        self.state = 216
                        self.match(QASMParser.T__10)
                        self.state = 217
                        self.match(QASMParser.T__17) 
                    self.state = 222
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,12,self._ctx)

                self.state = 223
                self.match(QASMParser.ID)
                self.state = 224
                self.match(QASMParser.T__9)
                self.state = 225
                self.match(QASMParser.INT)
                self.state = 226
                self.match(QASMParser.T__10)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 238
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,14,self._ctx)
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt==1:
                        self.state = 231
                        self._errHandler.sync(self)
                        _alt = self._interp.adaptivePredict(self._input,13,self._ctx)
                        while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                            if _alt==1:
                                self.state = 227
                                self.match(QASMParser.ID)
                                self.state = 228
                                self.match(QASMParser.T__17) 
                            self.state = 233
                            self._errHandler.sync(self)
                            _alt = self._interp.adaptivePredict(self._input,13,self._ctx)

                        self.state = 234
                        self.match(QASMParser.ID)
                        self.state = 235
                        self.match(QASMParser.T__17) 
                    self.state = 240
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,14,self._ctx)

                self.state = 241
                self.match(QASMParser.ID)
                self.state = 242
                self.match(QASMParser.T__9)
                self.state = 243
                self.match(QASMParser.INT)
                self.state = 244
                self.match(QASMParser.T__10)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ArgumentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(QASMParser.ID, 0)

        def INT(self):
            return self.getToken(QASMParser.INT, 0)

        def getRuleIndex(self):
            return QASMParser.RULE_argument

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterArgument" ):
                listener.enterArgument(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitArgument" ):
                listener.exitArgument(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitArgument" ):
                return visitor.visitArgument(self)
            else:
                return visitor.visitChildren(self)




    def argument(self):

        localctx = QASMParser.ArgumentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_argument)
        try:
            self.state = 252
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,16,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 247
                self.match(QASMParser.ID)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 248
                self.match(QASMParser.ID)
                self.state = 249
                self.match(QASMParser.T__9)
                self.state = 250
                self.match(QASMParser.INT)
                self.state = 251
                self.match(QASMParser.T__10)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExplistContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def exp(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(QASMParser.ExpContext)
            else:
                return self.getTypedRuleContext(QASMParser.ExpContext,i)


        def getRuleIndex(self):
            return QASMParser.RULE_explist

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExplist" ):
                listener.enterExplist(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExplist" ):
                listener.exitExplist(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExplist" ):
                return visitor.visitExplist(self)
            else:
                return visitor.visitChildren(self)




    def explist(self):

        localctx = QASMParser.ExplistContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_explist)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 259
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,17,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 254
                    self.exp(0)
                    self.state = 255
                    self.match(QASMParser.T__17) 
                self.state = 261
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,17,self._ctx)

            self.state = 262
            self.exp(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExpContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def REAL(self):
            return self.getToken(QASMParser.REAL, 0)

        def INT(self):
            return self.getToken(QASMParser.INT, 0)

        def ID(self):
            return self.getToken(QASMParser.ID, 0)

        def exp(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(QASMParser.ExpContext)
            else:
                return self.getTypedRuleContext(QASMParser.ExpContext,i)


        def unaryop(self):
            return self.getTypedRuleContext(QASMParser.UnaryopContext,0)


        def getRuleIndex(self):
            return QASMParser.RULE_exp

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExp" ):
                listener.enterExp(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExp" ):
                listener.exitExp(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExp" ):
                return visitor.visitExp(self)
            else:
                return visitor.visitChildren(self)



    def exp(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = QASMParser.ExpContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 38
        self.enterRecursionRule(localctx, 38, self.RULE_exp, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 280
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [QASMParser.REAL]:
                self.state = 265
                self.match(QASMParser.REAL)
                pass
            elif token in [QASMParser.INT]:
                self.state = 266
                self.match(QASMParser.INT)
                pass
            elif token in [QASMParser.T__18]:
                self.state = 267
                self.match(QASMParser.T__18)
                pass
            elif token in [QASMParser.ID]:
                self.state = 268
                self.match(QASMParser.ID)
                pass
            elif token in [QASMParser.T__20]:
                self.state = 269
                self.match(QASMParser.T__20)
                self.state = 270
                self.exp(4)
                pass
            elif token in [QASMParser.T__3]:
                self.state = 271
                self.match(QASMParser.T__3)
                self.state = 272
                self.exp(0)
                self.state = 273
                self.match(QASMParser.T__4)
                pass
            elif token in [QASMParser.T__24, QASMParser.T__25, QASMParser.T__26, QASMParser.T__27, QASMParser.T__28, QASMParser.T__29]:
                self.state = 275
                self.unaryop()
                self.state = 276
                self.match(QASMParser.T__3)
                self.state = 277
                self.exp(0)
                self.state = 278
                self.match(QASMParser.T__4)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 299
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,20,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 297
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,19,self._ctx)
                    if la_ == 1:
                        localctx = QASMParser.ExpContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_exp)
                        self.state = 282
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 8)")
                        self.state = 283
                        self.match(QASMParser.T__19)
                        self.state = 284
                        self.exp(9)
                        pass

                    elif la_ == 2:
                        localctx = QASMParser.ExpContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_exp)
                        self.state = 285
                        if not self.precpred(self._ctx, 7):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 7)")
                        self.state = 286
                        self.match(QASMParser.T__20)
                        self.state = 287
                        self.exp(8)
                        pass

                    elif la_ == 3:
                        localctx = QASMParser.ExpContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_exp)
                        self.state = 288
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 289
                        self.match(QASMParser.T__21)
                        self.state = 290
                        self.exp(7)
                        pass

                    elif la_ == 4:
                        localctx = QASMParser.ExpContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_exp)
                        self.state = 291
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 292
                        self.match(QASMParser.T__22)
                        self.state = 293
                        self.exp(6)
                        pass

                    elif la_ == 5:
                        localctx = QASMParser.ExpContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_exp)
                        self.state = 294
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 295
                        self.match(QASMParser.T__23)
                        self.state = 296
                        self.exp(4)
                        pass

             
                self.state = 301
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,20,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class UnaryopContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def sin(self):
            return self.getTypedRuleContext(QASMParser.SinContext,0)


        def cos(self):
            return self.getTypedRuleContext(QASMParser.CosContext,0)


        def tan(self):
            return self.getTypedRuleContext(QASMParser.TanContext,0)


        def expx(self):
            return self.getTypedRuleContext(QASMParser.ExpxContext,0)


        def ln(self):
            return self.getTypedRuleContext(QASMParser.LnContext,0)


        def sqrt(self):
            return self.getTypedRuleContext(QASMParser.SqrtContext,0)


        def getRuleIndex(self):
            return QASMParser.RULE_unaryop

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterUnaryop" ):
                listener.enterUnaryop(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitUnaryop" ):
                listener.exitUnaryop(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitUnaryop" ):
                return visitor.visitUnaryop(self)
            else:
                return visitor.visitChildren(self)




    def unaryop(self):

        localctx = QASMParser.UnaryopContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_unaryop)
        try:
            self.state = 308
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [QASMParser.T__24]:
                self.enterOuterAlt(localctx, 1)
                self.state = 302
                self.sin()
                pass
            elif token in [QASMParser.T__25]:
                self.enterOuterAlt(localctx, 2)
                self.state = 303
                self.cos()
                pass
            elif token in [QASMParser.T__26]:
                self.enterOuterAlt(localctx, 3)
                self.state = 304
                self.tan()
                pass
            elif token in [QASMParser.T__27]:
                self.enterOuterAlt(localctx, 4)
                self.state = 305
                self.expx()
                pass
            elif token in [QASMParser.T__28]:
                self.enterOuterAlt(localctx, 5)
                self.state = 306
                self.ln()
                pass
            elif token in [QASMParser.T__29]:
                self.enterOuterAlt(localctx, 6)
                self.state = 307
                self.sqrt()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SinContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return QASMParser.RULE_sin

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSin" ):
                listener.enterSin(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSin" ):
                listener.exitSin(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSin" ):
                return visitor.visitSin(self)
            else:
                return visitor.visitChildren(self)




    def sin(self):

        localctx = QASMParser.SinContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_sin)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 310
            self.match(QASMParser.T__24)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CosContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return QASMParser.RULE_cos

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCos" ):
                listener.enterCos(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCos" ):
                listener.exitCos(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCos" ):
                return visitor.visitCos(self)
            else:
                return visitor.visitChildren(self)




    def cos(self):

        localctx = QASMParser.CosContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_cos)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 312
            self.match(QASMParser.T__25)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TanContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return QASMParser.RULE_tan

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTan" ):
                listener.enterTan(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTan" ):
                listener.exitTan(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTan" ):
                return visitor.visitTan(self)
            else:
                return visitor.visitChildren(self)




    def tan(self):

        localctx = QASMParser.TanContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_tan)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 314
            self.match(QASMParser.T__26)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExpxContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return QASMParser.RULE_expx

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExpx" ):
                listener.enterExpx(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExpx" ):
                listener.exitExpx(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExpx" ):
                return visitor.visitExpx(self)
            else:
                return visitor.visitChildren(self)




    def expx(self):

        localctx = QASMParser.ExpxContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_expx)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 316
            self.match(QASMParser.T__27)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LnContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return QASMParser.RULE_ln

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLn" ):
                listener.enterLn(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLn" ):
                listener.exitLn(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLn" ):
                return visitor.visitLn(self)
            else:
                return visitor.visitChildren(self)




    def ln(self):

        localctx = QASMParser.LnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_ln)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 318
            self.match(QASMParser.T__28)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SqrtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return QASMParser.RULE_sqrt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSqrt" ):
                listener.enterSqrt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSqrt" ):
                listener.exitSqrt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSqrt" ):
                return visitor.visitSqrt(self)
            else:
                return visitor.visitChildren(self)




    def sqrt(self):

        localctx = QASMParser.SqrtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_sqrt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 320
            self.match(QASMParser.T__29)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[19] = self.exp_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def exp_sempred(self, localctx:ExpContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 8)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 7)
         

            if predIndex == 2:
                return self.precpred(self._ctx, 6)
         

            if predIndex == 3:
                return self.precpred(self._ctx, 5)
         

            if predIndex == 4:
                return self.precpred(self._ctx, 3)
         




