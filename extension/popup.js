const API_URL = "http://localhost:8000/predict";
const MAX_COMMENTS = 500;

const button = document.getElementById("analyze");
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const resultsEl = document.getElementById("results");

button.addEventListener("click", async () => {
  button.disabled = true;
  summaryEl.innerHTML = "";
  resultsEl.innerHTML = "";
  statusEl.textContent = "Reading comments from this page...";

  let comments = [];
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const response = await chrome.tabs.sendMessage(tab.id, { type: "GET_COMMENTS" });
    comments = (response && response.comments) || [];
  } catch (e) {
    statusEl.textContent = "Could not read this page. Refresh the YouTube tab, then try again.";
    button.disabled = false;
    return;
  }

  if (comments.length === 0) {
    statusEl.textContent = "No comments found yet. Scroll down on the video to load comments, then try again.";
    button.disabled = false;
    return;
  }

  comments = comments.slice(0, MAX_COMMENTS);
  statusEl.textContent = "Found " + comments.length + " comments. Analyzing...";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: comments.map((text) => ({ text })) }),
    });
    if (!res.ok) throw new Error("bad response");
    const data = await res.json();
    showResults(data.results || []);
  } catch (e) {
    statusEl.textContent = "Backend not reachable. Is the API running on localhost:8000?";
  }
  button.disabled = false;
});

function showResults(results) {
  const counts = { Positive: 0, Negative: 0, Neutral: 0 };
  results.forEach((r) => { if (r.sentiment in counts) counts[r.sentiment]++; });

  summaryEl.innerHTML =
    '<div class="chip pos">Positive<br>' + counts.Positive + "</div>" +
    '<div class="chip neg">Negative<br>' + counts.Negative + "</div>" +
    '<div class="chip neu">Neutral<br>' + counts.Neutral + "</div>";

  results.forEach((r) => {
    const li = document.createElement("li");
    const tag = document.createElement("span");
    tag.className = "tag " + r.sentiment;
    tag.textContent = r.sentiment + ": ";
    li.appendChild(tag);
    li.appendChild(document.createTextNode(r.text));
    resultsEl.appendChild(li);
  });

  statusEl.textContent = "Done - analyzed " + results.length + " comments.";
}
