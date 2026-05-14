// Gemma Teach — homework template
#let notes(title: "Homework", body) = {
  set page(margin: (x: 1in, y: 1in))
  set text(size: 11pt)
  set enum(numbering: "1.", spacing: 0.8em)
  show heading.where(level: 1): h => block(below: 0.6em, text(size: 18pt, weight: "bold", h.body))
  show heading.where(level: 2): h => block(above: 1em, below: 0.4em, text(size: 14pt, weight: "bold", h.body))
  align(center, text(size: 22pt, weight: "bold", title))
  v(0.4em)
  align(center, text(size: 10pt, fill: gray, "Homework — " + datetime.today().display("[year]-[month]-[day]")))
  v(1.2em)
  body
}
