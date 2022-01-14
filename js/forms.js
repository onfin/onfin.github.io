import {h, render, useState} from '/js/preact.js';

export
function makeForm(div, fields, calc) {
    const defaultValues = {};
    for (let k in fields) {
        defaultValues[k] = fields[k].default;
        if (!fields[k].type)
            fields[k].type = "number";
    }
    calc(defaultValues);
    const PForm = () => {
        const [values, setValues] = useState(defaultValues);
        const onChange = (name, value, e) => {
            if (fields[name].type == "number" && value == String(Number(value)))
              value = Number(value);
            const newValues = Object.assign({}, values, {[name]: value});
            calc(newValues);
            setValues(newValues);
        }
        const res = [];
        for (let k in fields) {
            if (!fields[k].output) {
                //res.push(h('div', null,
                   res.push(h('span', null, fields[k].title));
                   res.push(h('input',
                        Object.assign({}, fields[k],
                            {
                                id: k,
                                [fields[k].type == "checkbox" ? 'checked' : 'value']: values[k],
                                onChange: (e) => onChange(k,e.target[fields[k].type == "checkbox" ? 'checked' : 'value'], e)
                            })));
                //    )))));
            }
            else {
                //res.push(h('div', null,
                    res.push(h('span', null, fields[k].title));
                    res.push(h('input', Object.assign({}, fields[k], {value: values[k], readonly: true, class: "output"})));
                //));
            }
        }
        return res; //h('div', {}, res);
    }

    render(h(PForm), div);
}

export
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