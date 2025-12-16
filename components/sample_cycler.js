// Detect Streamlit theme from URL query params
function detectTheme() {
  const params = new URLSearchParams(window.location.search);
  const bgColor = params.get('backgroundColor');
  if (bgColor) {
    const rgb = parseInt(bgColor.slice(1), 16);
    const r = (rgb >> 16) & 255;
    const g = (rgb >> 8) & 255;
    const b = rgb & 255;
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return luminance < 0.5 ? 'dark' : 'light';
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

// Check if content needs truncation (more than ~8 lines)
function needsTruncation(element) {
  const lineHeight = parseFloat(getComputedStyle(element).lineHeight) || 24;
  return element.scrollHeight > lineHeight * 10;
}

// Sample cycling logic
function initSampleCycler(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  if (detectTheme() === 'dark') {
    container.classList.add('dark-theme');
  }

  const samples = container.querySelectorAll('.sample-content');
  const counter = container.querySelector('.sample-counter');
  const prevBtn = container.querySelector('.prev-btn');
  const nextBtn = container.querySelector('.next-btn');
  const hint = container.querySelector('.expand-hint');

  let currentIdx = 0;
  const total = samples.length;
  const expandedState = new Array(total).fill(false);

  function updateDisplay() {
    samples.forEach((sample, i) => {
      const isVisible = i === currentIdx;
      sample.style.display = isVisible ? 'block' : 'none';

      if (isVisible) {
        // Apply collapsed state if not expanded and content is long
        if (!expandedState[i] && needsTruncation(sample)) {
          sample.classList.add('collapsed');
          if (hint) hint.style.display = 'block';
          if (hint) hint.textContent = 'Click to expand';
        } else {
          sample.classList.remove('collapsed');
          if (hint && expandedState[i]) {
            hint.style.display = 'block';
            hint.textContent = 'Click to collapse';
          } else if (hint && !needsTruncation(sample)) {
            hint.style.display = 'none';
          }
        }
      }
    });
    if (counter) counter.textContent = `Sample ${currentIdx + 1} of ${total}`;
  }

  // Click to toggle expand/collapse
  samples.forEach((sample, i) => {
    sample.addEventListener('click', () => {
      expandedState[i] = !expandedState[i];
      updateDisplay();
    });
  });

  if (hint) {
    hint.addEventListener('click', () => {
      expandedState[currentIdx] = !expandedState[currentIdx];
      updateDisplay();
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      currentIdx = (currentIdx - 1 + total) % total;
      updateDisplay();
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      currentIdx = (currentIdx + 1) % total;
      updateDisplay();
    });
  }

  updateDisplay();
}
