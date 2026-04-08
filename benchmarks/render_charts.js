const fs = require('fs');
const path = require('path');

function latestMeasuredSnapshot(resultsDir) {
  return fs.readdirSync(resultsDir)
    .filter((name) => /^\d{4}-\d{2}-\d{2}.*\.json$/.test(name))
    .sort()
    .at(-1);
}

function escapeXml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function niceCeil(value) {
  if (value <= 1) return 1;
  const exponent = Math.floor(Math.log10(value));
  const fraction = value / Math.pow(10, exponent);
  let niceFraction = 10;
  if (fraction <= 1) niceFraction = 1;
  else if (fraction <= 2) niceFraction = 2;
  else if (fraction <= 5) niceFraction = 5;
  return niceFraction * Math.pow(10, exponent);
}

function barChartSvg({
  title,
  subtitle,
  yLabel,
  data,
  valueFormatter,
  yMax,
  color = '#1d4ed8',
  width = 860,
  height = 360,
}) {
  const margin = { top: 58, right: 24, bottom: 82, left: 70 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const maxValue = yMax ?? niceCeil(Math.max(...data.map((d) => d.value)) * 1.15);
  const stepCount = 5;
  const band = innerWidth / data.length;
  const barWidth = Math.min(56, band * 0.62);

  const grid = [];
  for (let i = 0; i <= stepCount; i++) {
    const value = (maxValue / stepCount) * i;
    const y = margin.top + innerHeight - (value / maxValue) * innerHeight;
    grid.push(`
      <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#e5e7eb" stroke-width="1" />
      <text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" class="tick">${escapeXml(valueFormatter(value, true))}</text>
    `);
  }

  const bars = data.map((item, index) => {
    const x = margin.left + band * index + (band - barWidth) / 2;
    const barHeight = (item.value / maxValue) * innerHeight;
    const y = margin.top + innerHeight - barHeight;
    const labelY = margin.top + innerHeight + 22;
    const secondaryY = labelY + 16;
    const placeValueInsideBar = item.valuePosition === 'inside'
      || (item.valuePosition !== 'outside' && y < margin.top + 22 && barHeight >= 28);
    const valueY = placeValueInsideBar ? y + 16 : y - 8;
    const fill = item.fill ?? color;
    const stroke = item.stroke ?? 'none';
    const strokeWidth = item.strokeWidth ?? 0;
    const valueFill = item.valueFill ?? (placeValueInsideBar ? '#ffffff' : '#0f172a');
    const labelFill = item.labelFill ?? '#0f172a';
    const secondaryFill = item.secondaryFill ?? '#475569';
    return `
      <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}" />
      <text x="${x + barWidth / 2}" y="${valueY}" text-anchor="middle" class="value" style="fill: ${valueFill};">${escapeXml(valueFormatter(item.value, false))}</text>
      <text x="${x + barWidth / 2}" y="${labelY}" text-anchor="middle" class="label" style="fill: ${labelFill};">${escapeXml(item.label)}</text>
      ${item.secondary ? `<text x="${x + barWidth / 2}" y="${secondaryY}" text-anchor="middle" class="secondary" style="fill: ${secondaryFill};">${escapeXml(item.secondary)}</text>` : ''}
    `;
  }).join('\n');

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeXml(title)}">
  <style>
    .title { font: 700 20px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f172a; }
    .subtitle { font: 400 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #475569; }
    .label { font: 600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f172a; }
    .secondary { font: 400 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #475569; }
    .tick { font: 400 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #64748b; }
    .value { font: 700 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f172a; }
    .axis { font: 600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #334155; }
  </style>
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="${margin.left}" y="28" class="title">${escapeXml(title)}</text>
  <text x="${margin.left}" y="46" class="subtitle">${escapeXml(subtitle)}</text>
  <text x="18" y="${margin.top + innerHeight / 2}" transform="rotate(-90, 18, ${margin.top + innerHeight / 2})" class="axis">${escapeXml(yLabel)}</text>
  ${grid.join('\n')}
  <line x1="${margin.left}" y1="${margin.top + innerHeight}" x2="${width - margin.right}" y2="${margin.top + innerHeight}" stroke="#94a3b8" stroke-width="1.5" />
  ${bars}
</svg>`;
}

function ratioVsBaselineText(value, baseline, suffix) {
  if (baseline <= 0) return '';
  const ratio = value / baseline;
  if (Math.abs(ratio - 1) < 1e-9) {
    return suffix;
  }
  return `${ratio.toFixed(1)}x ${suffix}`;
}

function highlightComparisonRows(rows, baselineLibrary, baselineSecondary, comparisonSuffix) {
  const rowName = (row) => row.library ?? row.label;
  const baseline = rows.find((row) => rowName(row) === baselineLibrary);
  if (!baseline) return rows;
  return rows.map((row) => {
    const isBaseline = rowName(row) === baselineLibrary;
    return {
      ...row,
      secondary: isBaseline
        ? baselineSecondary
        : ratioVsBaselineText(row.value, baseline.value, comparisonSuffix),
      fill: isBaseline ? '#635bff' : '#9aa7b6',
      stroke: isBaseline ? '#4338ca' : 'none',
      strokeWidth: isBaseline ? 1.5 : 0,
      valueFill: isBaseline ? '#4338ca' : '#0f172a',
      labelFill: isBaseline ? '#4338ca' : '#0f172a',
      secondaryFill: isBaseline ? '#4338ca' : '#64748b',
    };
  });
}

function writeChart(filePath, svg) {
  fs.writeFileSync(filePath, svg);
  console.log(`wrote ${path.relative(process.cwd(), filePath)}`);
}

function kiB(bytes) {
  return bytes / 1024;
}

function main() {
  const rootDir = path.resolve(__dirname, '..');
  const resultsDir = path.join(rootDir, 'benchmarks', 'results');
  const inputArg = process.argv[2];
  const outputArg = process.argv[3];
  const inputPath = inputArg
    ? path.resolve(process.cwd(), inputArg)
    : path.join(resultsDir, latestMeasuredSnapshot(resultsDir));
  const outputDir = outputArg
    ? path.resolve(process.cwd(), outputArg)
    : path.join(rootDir, 'benchmarks', 'charts');

  fs.mkdirSync(outputDir, { recursive: true });

  const snapshot = JSON.parse(fs.readFileSync(inputPath, 'utf8'));

  writeChart(
    path.join(outputDir, 'recall-vs-bits.svg'),
    barChartSvg({
      title: 'Recall@10 vs bit-width',
      subtitle: 'Synthetic clustered embeddings, d=384, N=5000',
      yLabel: 'Recall@10 (%)',
      yMax: 100,
      color: '#0f766e',
      data: snapshot.accuracy_vs_bits.map((row) => ({
        label: `${row.bits}b`,
        secondary: `${row.memory_bytes_per_vector} B/vec`,
        value: row.recall10_pct,
      })),
      valueFormatter: (value, tick) => tick ? `${value.toFixed(0)}%` : `${value.toFixed(1)}%`,
    })
  );

  writeChart(
    path.join(outputDir, 'search-vs-corpus.svg'),
    barChartSvg({
      title: 'Search latency vs corpus size',
      subtitle: '4-bit search on synthetic clustered embeddings, d=384',
      yLabel: 'Search time (ms)',
      color: '#1d4ed8',
      data: snapshot.corpus_scaling_4bit.map((row) => ({
        label: row.corpus_size.toLocaleString(),
        secondary: `${row.recall10_pct.toFixed(1)}% r@10`,
        value: row.search_ms,
      })),
      valueFormatter: (value, tick) => tick ? `${value.toFixed(0)}` : `${value.toFixed(2)} ms`,
    })
  );

  writeChart(
    path.join(outputDir, 'tail-latency-5k.svg'),
    barChartSvg({
      title: 'Tail latency under sustained load',
      subtitle: '200 queries, d=384, 4-bit, N=5000',
      yLabel: 'Latency (ms)',
      yMax: niceCeil(snapshot.sustained_load_5k_4bit.p99_ms * 1.15),
      color: '#7c3aed',
      data: [
        { label: 'p50', value: snapshot.sustained_load_5k_4bit.p50_ms },
        { label: 'p95', value: snapshot.sustained_load_5k_4bit.p95_ms },
        { label: 'p99', value: snapshot.sustained_load_5k_4bit.p99_ms },
      ],
      valueFormatter: (value, tick) => tick ? `${value.toFixed(0)}` : `${value.toFixed(2)} ms`,
    })
  );

  writeChart(
    path.join(outputDir, 'bundle-size.svg'),
    barChartSvg({
      title: 'Bundle size',
      subtitle: 'Measured on the generated web package with gzip -9',
      yLabel: 'Size (KiB)',
      color: '#b45309',
      data: [
        { label: 'wasm raw', value: kiB(snapshot.bundle_size.wasm_raw_bytes) },
        { label: 'wasm gzip', value: kiB(snapshot.bundle_size.wasm_gzip_bytes) },
        { label: 'js raw', value: kiB(snapshot.bundle_size.js_raw_bytes) },
        { label: 'js gzip', value: kiB(snapshot.bundle_size.js_gzip_bytes) },
        { label: 'total raw', value: kiB(snapshot.bundle_size.total_raw_bytes) },
        { label: 'total gzip', value: kiB(snapshot.bundle_size.total_gzip_bytes) },
      ],
      valueFormatter: (value, tick) => tick ? `${value.toFixed(0)}` : `${value.toFixed(1)} KiB`,
    })
  );

  const comparisonPath = path.join(resultsDir, 'analysis-derived-comparison.json');
  if (!fs.existsSync(comparisonPath)) {
    return;
  }

  const comparison = JSON.parse(fs.readFileSync(comparisonPath, 'utf8'));

  writeChart(
    path.join(outputDir, 'bundle-size-vs-alternatives.svg'),
    barChartSvg({
      title: 'Bundle size vs alternative browser-side search libraries',
      subtitle: 'Purple = current measured turboquant-wasm build. Gray bars = comparison estimates documented in benchmarks/wasm_analysis.md',
      yLabel: 'Total gzip size (KiB)',
      color: '#0f766e',
      data: highlightComparisonRows(
        comparison.bundle_size_gzip_kib.map((row) => ({
          label: row.library,
          secondary: row.secondary,
          value: row.value,
        })),
        'turboquant',
        'current build',
        'larger'
      ),
      valueFormatter: (value, tick) => tick ? `${value.toFixed(0)}` : `${value.toFixed(1)} KiB`,
    })
  );

  writeChart(
    path.join(outputDir, 'memory-d384-vs-alternatives.svg'),
    barChartSvg({
      title: 'Memory per vector at d=384',
      subtitle: 'Purple = current measured packed TurboQuant storage. Gray bars = comparison estimates documented in benchmarks/wasm_analysis.md',
      yLabel: 'Bytes per vector',
      color: '#1d4ed8',
      data: highlightComparisonRows(
        comparison.memory_per_vector_bytes.d384_4bit.map((row) => ({
          label: row.library,
          secondary: row.secondary,
          value: row.value,
        })),
        'turboquant',
        'current packed',
        'more memory'
      ),
      valueFormatter: (value, tick) => tick ? `${value.toFixed(0)}` : `${value.toFixed(0)} B`,
    })
  );

  writeChart(
    path.join(outputDir, 'memory-d1536-vs-alternatives.svg'),
    barChartSvg({
      title: 'Memory per vector at d=1536',
      subtitle: 'Purple = current measured packed TurboQuant storage. Gray bars = comparison estimates documented in benchmarks/wasm_analysis.md',
      yLabel: 'Bytes per vector',
      color: '#7c3aed',
      data: highlightComparisonRows(
        comparison.memory_per_vector_bytes.d1536_4bit.map((row) => ({
          label: row.library,
          secondary: row.secondary,
          value: row.value,
        })),
        'turboquant',
        'current packed',
        'more memory'
      ),
      valueFormatter: (value, tick) => tick ? `${value.toFixed(0)}` : `${value.toFixed(0)} B`,
    })
  );
}

main();
