const generalization_fr_examples = document.querySelectorAll('.generalization_fr');
const cross_page = document.querySelectorAll('.cross_page');
generalization_fr_examples[0].classList.add('active');
cross_page[0].classList.add('active');
cross_page.forEach((page) => {
  page.addEventListener('click', () => {
    const pageNumber = page.getAttribute('data-page');

    // Hide all examples
    generalization_fr_examples.forEach((example) => {
      example.classList.remove('active');
    });
    cross_page.forEach((page) => {
      page.classList.remove('active');
    });

    generalization_fr_examples[pageNumber-1].classList.add('active');
    cross_page[pageNumber-1].classList.add('active');

  });
});


const transcriptsCache = {};

async function loadTranscripts(testset) {
  if (transcriptsCache[testset]) {
    return transcriptsCache[testset];
  }
  const response = await fetch(`./audio_ready/MR/${testset}/text`);
  const textData = await response.text();

  const transcripts = {};
  textData.split("\n").forEach(line => {
    const [id, ...rest] = line.split(" ");
    if (id) transcripts[id] = rest.join(" ");
  });

  transcriptsCache[testset] = transcripts;
  return transcripts;
}

async function preloadAllTranscripts() {
  const utterances = document.querySelectorAll(".utterance");
  const testsets = new Set();
  utterances.forEach(u => testsets.add(u.dataset.testset));

  await Promise.all([...testsets].map(ts => loadTranscripts(ts)));
  console.log("All transcripts preloaded:", Object.keys(transcriptsCache));
}

document.querySelectorAll(".utterance").forEach(utterance => {
  const utt_id = utterance.dataset.uttId;
  const testset = utterance.dataset.testset;
  const transcriptSpans = utterance.querySelectorAll(".transcript");
  const details = utterance.querySelector("details");

  details.addEventListener("toggle", async () => {
    if (details.open) {
      const transcripts = await loadTranscripts(testset);
      const text = transcripts[utt_id] || "[Transcript not found]";
      transcriptSpans.forEach(span => {
        if (!span.textContent) {
          span.textContent = text;
        }
      });
    }
  });

  utterance.querySelectorAll("audio[data-codec]").forEach(audio => {
    const codec = audio.getAttribute("data-codec");
    audio.src = `./audio_ready/MR/${testset}/${codec}/${utt_id}.wav`;
  });
});
