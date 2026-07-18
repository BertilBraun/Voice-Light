export class SpokenTextProgress {
  constructor() {
    this.text = "";
    this.spokenOffset = 0;
    this.acknowledgedOffset = 0;
  }

  replaceText(text) {
    this.text = text;
  }

  appendText(text) {
    this.text += text;
  }

  markSpoken(offset) {
    this.spokenOffset = Math.max(this.spokenOffset, offset);
  }

  acknowledge(offset) {
    this.acknowledgedOffset = Math.max(this.acknowledgedOffset, offset);
  }

  settleInterruptedText() {
    this.spokenOffset = this.acknowledgedOffset;
  }

  spokenText() {
    return sliceTextByCharacterOffset(this.text, 0, this.spokenOffset);
  }

  unspokenText() {
    return sliceTextByCharacterOffset(this.text, this.spokenOffset);
  }
}

function sliceTextByCharacterOffset(text, start, end) {
  return Array.from(text).slice(start, end).join("");
}
