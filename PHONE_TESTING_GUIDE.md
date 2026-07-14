# 📱 Testing Personal Assistant AI on Your Phone

Complete guide to access the web interface from your phone via local WiFi.

---

## ✅ Prerequisites

- ✓ Data pipeline ran successfully (`data/processed/chunks.jsonl` exists)
- ✓ Streamlit installed (`pip install streamlit`)
- ✓ Phone on same WiFi network as computer
- ✓ ~5 minutes of setup time

---

## 🚀 Step 1: Get Your Computer's IP Address

Your phone needs to know which computer to connect to.

### On Linux/Mac (Terminal):
```bash
# Method 1: Get WiFi IP
ifconfig | grep "inet " | grep -v 127.0.0.1

# Method 2: Direct command
python -c "import socket; print(socket.gethostbyname(socket.gethostname()))"
```

Expected output: `192.168.x.x` or `10.0.x.x`

### On Windows (Command Prompt):
```bash
ipconfig

# Look for "IPv4 Address" under your WiFi adapter
```

**Example**: `192.168.1.45`

---

## 🌐 Step 2: Start the Streamlit Server

Run from your computer (in the project directory):

```bash
cd /home/user/TrikRide_App

streamlit run src/interface/app.py --server.address 0.0.0.0
```

**Expected output**:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.45:8501
```

✅ Note the **Network URL** - this is what you'll use on your phone!

---

## 📱 Step 3: Open on Your Phone

### On Your Phone's Browser:

1. **Open Safari, Chrome, or Firefox**
2. **Enter the Network URL** from Step 2
   - Example: `http://192.168.1.45:8501`
3. **Press Enter**

You should see the Personal Assistant AI interface load! 🎉

---

## 💬 Step 4: Test It Out

### What You Can Do:

1. **View Dataset Info** (Right sidebar):
   - Number of documents indexed
   - Total chunks and tokens
   - List of documents

2. **Ask Questions** (Main chat area):
   - Type: "What is RAG?"
   - Type: "What are the semester deadlines?"
   - Type: "Tell me about embeddings"

3. **Adjust Settings** (Sidebar):
   - Change "Number of results" (1-10)
   - Adjust "Min similarity" threshold (0.0-1.0)

4. **View Results**:
   - See relevant chunks ranked by similarity
   - Read source document information
   - View full chunk text in expandable sections

### Example Queries to Try:

```
1. "What is a capstone project?"
2. "When is the final exam?"
3. "Explain vector databases"
4. "How do embeddings work?"
5. "What are the main deliverables?"
```

---

## 🔧 Troubleshooting

### "Cannot connect to http://192.168.1.45:8501"

**Problem**: Phone not connecting to server

**Solutions**:
1. Check phone is on same WiFi network
2. Try computer's IP again (run `ifconfig` again)
3. Make sure Streamlit is still running on computer
4. Check firewall isn't blocking port 8501

### "Connection refused" or "ERR_CONNECTION_REFUSED"

**Problem**: Wrong IP address

**Solution**:
- Use Network URL from Streamlit startup message
- Not the local URL (localhost won't work from phone)

### "No processed data found"

**Problem**: Pipeline hasn't been run yet

**Solution**:
```bash
python src/pipeline.py
```

Then refresh phone browser.

### Streamlit is slow on phone

**Normal**: First load takes a few seconds (model initialization)  
**Solutions**:
- Reload page
- Reduce "Number of results" in settings
- Increase "Min similarity" threshold

---

## 🎯 What's Happening Behind the Scenes

When you ask a question on your phone:

1. **Your question is embedded** using the same model as documents
2. **Similarity search** finds the most relevant document chunks
3. **Results are ranked** by cosine similarity score
4. **Top results are displayed** with full context

This is **Retrieval-Augmented Generation (RAG)** in action!

---

## 📊 Example Interaction

**You ask**: "What is the capstone deadline?"

**System finds**:
- Chunk 1: `schedule_semester_2026.txt` - Similarity: 87%
  - "Capstone Checkpoint 1 (Prelim): August 15, 2026, 11:59 PM"
- Chunk 2: `project_capstone_plan.txt` - Similarity: 72%
  - "Due dates by checkpoint..."
- Chunk 3: `task_list_sprint1.txt` - Similarity: 65%
  - "Submit Checkpoint 1 by August 15"

---

## 🔄 Keeping It Running

The server keeps running as long as:
- Computer is on
- Streamlit terminal is open
- WiFi stays connected

To stop:
- Press `Ctrl+C` in the terminal where Streamlit is running

To restart:
- Run the command again from Step 2

---

## 📈 Next Steps

After testing the semantic search:

1. **Add more documents** to `data/raw/` and rerun pipeline
2. **Build out Checkpoint 2**: Vector database, prompt engineering
3. **Deploy to cloud** for public access (when ready)
4. **Add conversation memory** for multi-turn Q&A
5. **Integrate with LLM** for AI-generated responses

---

## 💾 Notes

- Chat history is **stored in your session** (refreshing clears it)
- No data is sent to external servers (everything runs locally)
- Results are based on **semantic similarity**, not exact keyword matching
- Similarity threshold can be adjusted for precision vs recall

---

## 🎓 Learning Points

By using this interface, you're seeing:

✅ **Text Preprocessing**: Clean, normalized documents  
✅ **Embeddings**: Vector representations of text  
✅ **Similarity Search**: Finding relevant content  
✅ **RAG Workflow**: Retrieval before generation  

This is exactly what your capstone project demonstrates!

---

## 📞 Support

If something doesn't work:

1. Check prerequisites above
2. Ensure data pipeline ran: `ls -la data/processed/chunks.jsonl`
3. Verify Streamlit output shows Network URL
4. Try refreshing phone browser
5. Restart Streamlit server

---

**Ready to test?** 🚀

```bash
streamlit run src/interface/app.py --server.address 0.0.0.0
```

Then open the Network URL on your phone!
