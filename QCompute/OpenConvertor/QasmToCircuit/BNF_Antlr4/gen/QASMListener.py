# Generated from C:/Project/Quantum/QCompute/QCompute/OpenConvertor/QasmToCircuit/BNF_Antlr4\QASM.g4 by ANTLR 4.10.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .QASMParser import QASMParser
else:
    from QASMParser import QASMParser

# This class defines a complete listener for a parse tree produced by QASMParser.
class QASMListener(ParseTreeListener):

    # Enter a parse tree produced by QASMParser#mainprog.
    def enterMainprog(self, ctx:QASMParser.MainprogContext):
        pass

    # Exit a parse tree produced by QASMParser#mainprog.
    def exitMainprog(self, ctx:QASMParser.MainprogContext):
        pass


    # Enter a parse tree produced by QASMParser#statement.
    def enterStatement(self, ctx:QASMParser.StatementContext):
        pass

    # Exit a parse tree produced by QASMParser#statement.
    def exitStatement(self, ctx:QASMParser.StatementContext):
        pass


    # Enter a parse tree produced by QASMParser#version.
    def enterVersion(self, ctx:QASMParser.VersionContext):
        pass

    # Exit a parse tree produced by QASMParser#version.
    def exitVersion(self, ctx:QASMParser.VersionContext):
        pass


    # Enter a parse tree produced by QASMParser#include.
    def enterInclude(self, ctx:QASMParser.IncludeContext):
        pass

    # Exit a parse tree produced by QASMParser#include.
    def exitInclude(self, ctx:QASMParser.IncludeContext):
        pass


    # Enter a parse tree produced by QASMParser#qReg.
    def enterQReg(self, ctx:QASMParser.QRegContext):
        pass

    # Exit a parse tree produced by QASMParser#qReg.
    def exitQReg(self, ctx:QASMParser.QRegContext):
        pass


    # Enter a parse tree produced by QASMParser#cReg.
    def enterCReg(self, ctx:QASMParser.CRegContext):
        pass

    # Exit a parse tree produced by QASMParser#cReg.
    def exitCReg(self, ctx:QASMParser.CRegContext):
        pass


    # Enter a parse tree produced by QASMParser#gatedecl.
    def enterGatedecl(self, ctx:QASMParser.GatedeclContext):
        pass

    # Exit a parse tree produced by QASMParser#gatedecl.
    def exitGatedecl(self, ctx:QASMParser.GatedeclContext):
        pass


    # Enter a parse tree produced by QASMParser#goplist.
    def enterGoplist(self, ctx:QASMParser.GoplistContext):
        pass

    # Exit a parse tree produced by QASMParser#goplist.
    def exitGoplist(self, ctx:QASMParser.GoplistContext):
        pass


    # Enter a parse tree produced by QASMParser#qop.
    def enterQop(self, ctx:QASMParser.QopContext):
        pass

    # Exit a parse tree produced by QASMParser#qop.
    def exitQop(self, ctx:QASMParser.QopContext):
        pass


    # Enter a parse tree produced by QASMParser#uop.
    def enterUop(self, ctx:QASMParser.UopContext):
        pass

    # Exit a parse tree produced by QASMParser#uop.
    def exitUop(self, ctx:QASMParser.UopContext):
        pass


    # Enter a parse tree produced by QASMParser#comments.
    def enterComments(self, ctx:QASMParser.CommentsContext):
        pass

    # Exit a parse tree produced by QASMParser#comments.
    def exitComments(self, ctx:QASMParser.CommentsContext):
        pass


    # Enter a parse tree produced by QASMParser#lineComment.
    def enterLineComment(self, ctx:QASMParser.LineCommentContext):
        pass

    # Exit a parse tree produced by QASMParser#lineComment.
    def exitLineComment(self, ctx:QASMParser.LineCommentContext):
        pass


    # Enter a parse tree produced by QASMParser#multipleComments.
    def enterMultipleComments(self, ctx:QASMParser.MultipleCommentsContext):
        pass

    # Exit a parse tree produced by QASMParser#multipleComments.
    def exitMultipleComments(self, ctx:QASMParser.MultipleCommentsContext):
        pass


    # Enter a parse tree produced by QASMParser#measureOp.
    def enterMeasureOp(self, ctx:QASMParser.MeasureOpContext):
        pass

    # Exit a parse tree produced by QASMParser#measureOp.
    def exitMeasureOp(self, ctx:QASMParser.MeasureOpContext):
        pass


    # Enter a parse tree produced by QASMParser#barrierOp.
    def enterBarrierOp(self, ctx:QASMParser.BarrierOpContext):
        pass

    # Exit a parse tree produced by QASMParser#barrierOp.
    def exitBarrierOp(self, ctx:QASMParser.BarrierOpContext):
        pass


    # Enter a parse tree produced by QASMParser#anylist.
    def enterAnylist(self, ctx:QASMParser.AnylistContext):
        pass

    # Exit a parse tree produced by QASMParser#anylist.
    def exitAnylist(self, ctx:QASMParser.AnylistContext):
        pass


    # Enter a parse tree produced by QASMParser#idlist.
    def enterIdlist(self, ctx:QASMParser.IdlistContext):
        pass

    # Exit a parse tree produced by QASMParser#idlist.
    def exitIdlist(self, ctx:QASMParser.IdlistContext):
        pass


    # Enter a parse tree produced by QASMParser#mixedlist.
    def enterMixedlist(self, ctx:QASMParser.MixedlistContext):
        pass

    # Exit a parse tree produced by QASMParser#mixedlist.
    def exitMixedlist(self, ctx:QASMParser.MixedlistContext):
        pass


    # Enter a parse tree produced by QASMParser#argument.
    def enterArgument(self, ctx:QASMParser.ArgumentContext):
        pass

    # Exit a parse tree produced by QASMParser#argument.
    def exitArgument(self, ctx:QASMParser.ArgumentContext):
        pass


    # Enter a parse tree produced by QASMParser#explist.
    def enterExplist(self, ctx:QASMParser.ExplistContext):
        pass

    # Exit a parse tree produced by QASMParser#explist.
    def exitExplist(self, ctx:QASMParser.ExplistContext):
        pass


    # Enter a parse tree produced by QASMParser#exp.
    def enterExp(self, ctx:QASMParser.ExpContext):
        pass

    # Exit a parse tree produced by QASMParser#exp.
    def exitExp(self, ctx:QASMParser.ExpContext):
        pass


    # Enter a parse tree produced by QASMParser#unaryop.
    def enterUnaryop(self, ctx:QASMParser.UnaryopContext):
        pass

    # Exit a parse tree produced by QASMParser#unaryop.
    def exitUnaryop(self, ctx:QASMParser.UnaryopContext):
        pass


    # Enter a parse tree produced by QASMParser#sin.
    def enterSin(self, ctx:QASMParser.SinContext):
        pass

    # Exit a parse tree produced by QASMParser#sin.
    def exitSin(self, ctx:QASMParser.SinContext):
        pass


    # Enter a parse tree produced by QASMParser#cos.
    def enterCos(self, ctx:QASMParser.CosContext):
        pass

    # Exit a parse tree produced by QASMParser#cos.
    def exitCos(self, ctx:QASMParser.CosContext):
        pass


    # Enter a parse tree produced by QASMParser#tan.
    def enterTan(self, ctx:QASMParser.TanContext):
        pass

    # Exit a parse tree produced by QASMParser#tan.
    def exitTan(self, ctx:QASMParser.TanContext):
        pass


    # Enter a parse tree produced by QASMParser#expx.
    def enterExpx(self, ctx:QASMParser.ExpxContext):
        pass

    # Exit a parse tree produced by QASMParser#expx.
    def exitExpx(self, ctx:QASMParser.ExpxContext):
        pass


    # Enter a parse tree produced by QASMParser#ln.
    def enterLn(self, ctx:QASMParser.LnContext):
        pass

    # Exit a parse tree produced by QASMParser#ln.
    def exitLn(self, ctx:QASMParser.LnContext):
        pass


    # Enter a parse tree produced by QASMParser#sqrt.
    def enterSqrt(self, ctx:QASMParser.SqrtContext):
        pass

    # Exit a parse tree produced by QASMParser#sqrt.
    def exitSqrt(self, ctx:QASMParser.SqrtContext):
        pass



del QASMParser