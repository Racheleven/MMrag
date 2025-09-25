import argparse
from agents.document_processor import DocumentProcessor
from agents.search_agent import SearchAgent
from agents.qa_agent import QAAgent

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MMRAG系统')
    parser.add_argument('--mode', choices=['process', 'query'], required=True,
                      help='运行模式：process-处理文档，query-查询问答')
    parser.add_argument('--question', type=str,
                      help='查询问题（仅在query模式下需要）')
    args = parser.parse_args()

    if args.mode == 'process':
        # 处理文档模式
        processor = DocumentProcessor()
        processor.process_documents()
    else:
        # 查询问答模式
        if not args.question:
            print("错误：在query模式下必须提供问题")
            return
        
        search_agent = SearchAgent()
        qa_agent = QAAgent(search_agent)
        
        print(f"\n问题: {args.question}")
        answer = qa_agent.run(args.question)
        print("\n答案:", answer)

if __name__ == "__main__":
    main()
