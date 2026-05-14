// Gemma Teach — notes template
// Usage from a flow-generated entry.typ:
//   #import "<this>" as template
//   #show: template.notes.with(title: "Lesson")
//   <markdown-rendered-as-typst>

#let notes(title: "Lesson", body) = {
  set page(margin: (x: 1in, y: 1in))
  set text(size: 11pt)
  show heading.where(level: 1): h => block(below: 0.6em, text(size: 18pt, weight: "bold", h.body))
  show heading.where(level: 2): h => block(above: 1em, below: 0.4em, text(size: 14pt, weight: "bold", h.body))
  show heading.where(level: 3): h => block(above: 0.8em, below: 0.3em, text(size: 12pt, weight: "bold", h.body))
  align(center, text(size: 22pt, weight: "bold", title))
  v(0.6em)
  align(center, text(size: 10pt, fill: gray, datetime.today().display("[year]-[month]-[day]")))
  v(1.4em)
  body
}
