function scrapeComments() {
  const nodes = document.querySelectorAll("ytd-comment-thread-renderer #content-text");
  const texts = [];
  nodes.forEach((n) => {
    const t = n.innerText.trim();
    if (t) texts.push(t);
  });
  return texts;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "GET_COMMENTS") {
    sendResponse({ comments: scrapeComments() });
  }
});
