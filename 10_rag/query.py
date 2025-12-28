from graph import app

print("💬 Ask questions about your resume (type 'exit' to quit)\n")

while True:
    query = input("> ")
    if query.lower() in ["exit", "quit"]:
        break

    result = app.invoke({"query": query})

    print("\n🤖 Answer:")
    print(result["answer"])
    print("-" * 50)
