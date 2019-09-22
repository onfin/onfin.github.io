function Form(name, fields, calc) {
    this.form = document.getElementById(name);
    this.fields = fields;
    for (let k in this.fields)
        if (this.form.elements[k].type === "checkbox")
          this.form.elements[k].checked = this.fields[k];
        else
          this.form.elements[k].value = this.fields[k];
    this.calc = calc;

    this.onChange = function (e) {
      if (e.type === "checkbox")
        this.fields[e.id] = e.checked;
      else
        this.fields[e.id] = +e.value;
      this.calc();
    };

    this.calc();
  }