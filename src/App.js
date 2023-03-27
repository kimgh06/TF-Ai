import axios from "axios";
import React, { useState } from "react";

function App() {
  const [req, setReq] = useState("");
  async function request() {
    var url = "https://api.openai.com/v1/engines/text-davinci-002/completions";
    let open_ai_response;
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url);

    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.setRequestHeader("Authorization", "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");

    xhr.onreadystatechange = function () {
      if (xhr.readyState === 4) {
        console.log(xhr.status);
        console.log(xhr.responseText);
        open_ai_response = xhr.responseText;
        console.log(open_ai_response);
      }
    };

    var data = `{
    "prompt": ${req},
    "temperature": 0.7,
    "max_tokens": 256,
    "top_p": 1,
    "frequency_penalty": 0.75,
    "presence_penalty": 0
  }`;

    xhr.send(data);
  }
  return (
    <div className="App">
      <input className="input" value={req} onChange={e => setReq(e.target.value)}></input>
      <button onClick={request}>request</button>
    </div>
  );
}

export default App;
