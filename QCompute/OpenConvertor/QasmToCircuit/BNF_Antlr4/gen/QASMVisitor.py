# Generated from C:/Project/Quantum/QCompute/QCompute/OpenConvertor/QasmToCircuit/BNF_Antlr4\QASM.g4 by ANTLR 4.10.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .QASMParser import QASMParser
else:
    from QASMParser import QASMParser

# This class defines a complete generic visitor for a parse tree produced by QASMParser.

class QASMVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by QASMParser#mainprog.
    def visitMainprog(self, ctx:QASMParser.MainprogContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#statement.
    def visitStatement(self, ctx:QASMParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#version.
    def visitVersion(self, ctx:QASMParser.VersionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#include.
    def visitInclude(self, ctx:QASMParser.IncludeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#qReg.
    def visitQReg(self, ctx:QASMParser.QRegContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#cReg.
    def visitCReg(self, ctx:QASMParser.CRegContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#gatedecl.
    def visitGatedecl(self, ctx:QASMParser.GatedeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#goplist.
    def visitGoplist(self, ctx:QASMParser.GoplistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#qop.
    def visitQop(self, ctx:QASMParser.QopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#uop.
    def visitUop(self, ctx:QASMParser.UopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#comments.
    def visitComments(self, ctx:QASMParser.CommentsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#lineComment.
    def visitLineComment(self, ctx:QASMParser.LineCommentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#multipleComments.
    def visitMultipleComments(self, ctx:QASMParser.MultipleCommentsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#measureOp.
    def visitMeasureOp(self, ctx:QASMParser.MeasureOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#barrierOp.
    def visitBarrierOp(self, ctx:QASMParser.BarrierOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#anylist.
    def visitAnylist(self, ctx:QASMParser.AnylistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#idlist.
    def visitIdlist(self, ctx:QASMParser.IdlistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#mixedlist.
    def visitMixedlist(self, ctx:QASMParser.MixedlistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#argument.
    def visitArgument(self, ctx:QASMParser.ArgumentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#explist.
    def visitExplist(self, ctx:QASMParser.ExplistContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#exp.
    def visitExp(self, ctx:QASMParser.ExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#unaryop.
    def visitUnaryop(self, ctx:QASMParser.UnaryopContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#sin.
    def visitSin(self, ctx:QASMParser.SinContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#cos.
    def visitCos(self, ctx:QASMParser.CosContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#tan.
    def visitTan(self, ctx:QASMParser.TanContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#expx.
    def visitExpx(self, ctx:QASMParser.ExpxContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#ln.
    def visitLn(self, ctx:QASMParser.LnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by QASMParser#sqrt.
    def visitSqrt(self, ctx:QASMParser.SqrtContext):
        return self.visitChildren(ctx)



del QASMParser