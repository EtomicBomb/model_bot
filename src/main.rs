use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;

fn main() {
    let old_model = Model::random(Complexity::Moderate);
    println!("####################\n{:#?}\n#####################", old_model);
    let dataset = DataSet::from_model(&old_model);


    let mut model = Model::simple_model(&dataset);
    println!("{:#?}", model);

    for _ in 0..500 {
        let old_error = model.mean_squared_error(&dataset);
        let old_model = model.clone();
        model.improve(&dataset);
        let new_error = model.mean_squared_error(&dataset);
        if old_error < new_error {
            //println!("*{:#?}\n*\n{:#?}*\n", old_model, model);
        }
        model.prune(&dataset);

        println!("error {}", model.mean_squared_error(&dataset));
    }


    println!("\n\n{:#?}", model);
}


#[derive(Debug, Clone)]
struct Model {
    label: ModelLabel,
    children: Vec<Model>,
}

impl Model {
    fn prune(&mut self, dataset: &DataSet) {
        // this function just makes the argument that if a model equals one of it's submodels, it can just be
        // replaced with that submodel
        loop {
            let mut replace_with = None;

            'addr: for addr in self.get_all_addresses().into_iter() {
                let model = self.get(addr.clone());

                if model.label.arity() == 0 { continue } // models with arity 2 have no submodels

                // lets see if a model returns the same thing as one of its children
                for (i, child) in model.children.iter().enumerate() {
                    if model.is_equal(child, dataset) {
                        // lets just replace this whole model with the child
                        replace_with = Some((addr, i));
                        break 'addr;
                    }
                }
            }

            if let Some((addr, child_index)) = replace_with {
                let child_address = addr.clone().append_to_bottom(child_index);

                *self.get_mut(addr.clone()) = self.get(child_address.clone()).clone();
            } else {
                return;
            }
        }
    }

    fn is_equal(&self, other: &Model, dataset: &DataSet) -> bool {
        // we want to compare these two models and see if they return exactly the same thing under
        // a particular dataset. We don't worry about the y's of the dataset; we only care about x's

        let mut max_difference = 0.0;

        for &DataPoint { x, .. } in dataset.points.iter() {

            let y1 = self.evaluate(x).unwrap();
            let y2 = other.evaluate(x).unwrap();

            let difference = (y1-y2).abs();

            if difference > max_difference {
                max_difference = difference;
            }
        }

        max_difference < 0.01
    }

    fn improve(&mut self, dataset: &DataSet) {
        // we will select a random submodule to be replaced with an operation, and a new simple_model.
        let all_child_models = self.get_all_addresses();
        let address_to_fix: Address = all_child_models.choose(&mut thread_rng()).unwrap().clone();

        let place = self.get_mut(address_to_fix.clone());
        // choose a random label
        let label =
            if thread_rng().gen_bool(0.5) { ModelLabel::Add } else { ModelLabel::Multiply };

        *place = Model {
            label,
            children: vec![
                place.clone(),
                Model { label: ModelLabel::Constant(255.0), children: vec![] },
            ]
        };

        let submodel_address = address_to_fix.clone().append_to_bottom(1);

        self.improve_replace_simple_model(submodel_address, dataset);
    }

    fn improve_replace_simple_model(&mut self, mut addr: Address, dataset: &DataSet) {
        let mut new_dataset = DataSet::default();

        // we need to figure out all of the x and y values for the address
        for point in dataset.points.iter() {
            let new_y = self.inverse_evaluate(addr.clone(), point.x, point.y);
            new_dataset.add_point(point.x, new_y);
        }

        let new_model = Model::simple_model(&new_dataset);

        let old_model = self.get(addr.clone());

        if new_model.mean_squared_error(dataset) < old_model.mean_squared_error(dataset) {
            // then the new one is better

            *self.get_mut(addr) = new_model;

        } // else, we just keep it the same and do nothing
    }

    fn inverse_evaluate(&self, mut addr: Address, x: f64, y: f64) -> f64 {

        match addr.pop() {
            Some(i) => {
                let other_i = if i == 0 { 1 } else { 0 };
                let other_value = || self.children[other_i].evaluate(x).unwrap();

                match self.label {
                    ModelLabel::Add => {
                        self.children[i].inverse_evaluate(addr, x, y-other_value())
                    },
                    ModelLabel::Multiply => {
                        let o = other_value();
                        let input_y = if o == 0.0 { 0.0 } else { y/o };
                        self.children[i].inverse_evaluate(addr, x, input_y)
                    },
                    ModelLabel::Negative => self.children[i].inverse_evaluate(addr, x, -y),
                    ModelLabel::Divide => {
                        if i == 0 {
                            self.children[i].inverse_evaluate(addr, x, y*other_value())
                        } else {
                            self.children[i].inverse_evaluate(addr, x, other_value()/y)
                        }
                    },
                    ModelLabel::X | ModelLabel::Constant(_) | ModelLabel::Dummy =>  panic!("Can't index a model with arity 0!"),
                }
            },
            None => y,
        }
    }


    fn random(complexity: Complexity) -> Model {
        // return a random model

        let label = ModelLabel::random(complexity);
        let n_children = label.arity();
        let mut children = Vec::new();

        for _ in 0..n_children {
            let child_complexity = complexity.decrement_probably();
            children.push(Model::random(child_complexity));
        }

        Model { label, children }
    }

    fn simple_model(data_set: &DataSet) -> Model {
        // a simple model is a model with a depth of 1
        // returns a simple model that matches this data set well
        // there are two simple models currently: X and Constant

        // first we will check X
        let x_model = Model {
            label: ModelLabel::X,
            children: vec![],
        };

        let model_x_error = x_model.mean_squared_error(data_set);

        // now we have to do Constant
        // lets just take the average of all of the y's
        let y_sum: f64 = data_set.points.iter().map(|point| point.y).sum();
        let data_length = data_set.points.iter().count();
        let y_average = y_sum / data_length as f64;

        let constant_model = Model {
            label: ModelLabel::Constant(y_average),
            children: vec![],
        };

        let model_constant_error = constant_model.mean_squared_error(data_set);

        if model_x_error < model_constant_error {
            x_model
        } else {
            constant_model
        }
    }

    fn evaluate(&self, x: f64) -> Option<f64> {
        let child1 = || self.children[0].evaluate(x);
        let child2 = || self.children[1].evaluate(x);

        Some(match self.label {
            ModelLabel::Add => child1()? + child2()?,
            ModelLabel::Multiply => child1()? * child2()?,
            ModelLabel::Negative => -child1()?,
            ModelLabel::Divide => if child2()? == 0.0 { return None } else { child1()? / child2()? },
            ModelLabel::X => x,
            ModelLabel::Constant(c) => c,
            ModelLabel::Dummy => panic!("Can't evaluate a dummy label."),
        })
    }

    fn mean_squared_error(&self, data_set: &DataSet) -> f64 {
        let mut error_sum = 0.0;
        let mut data_length = 0;

        for data_point in data_set.points.iter() {
            let x = data_point.x;
            let real_y = data_point.y;

            let calculated_y = match self.evaluate(x) {
                Some(y) => y,
                None => continue, // we'll just ignore this data point if it causes problems for us
            };

            let error = real_y - calculated_y;
            error_sum += error*error;

            data_length += 1;
        }

        error_sum / data_length as f64
    }

    fn get(&self, mut addr: Address) -> &Model {
        match addr.pop() {
            Some(i) => self.children[i].get(addr),
            None => self,
        }
    }

    fn get_mut(&mut self, mut addr: Address) -> &mut Model {
        match addr.pop() {
            Some(i) => self.children[i].get_mut(addr),
            None => self,
        }
    }

    fn get_all_addresses(&self) -> Vec<Address> {
        let mut addresses = vec![Address::default()];

        for (i, child) in self.children.iter().enumerate() {
            let children_addresses_iter = child.get_all_addresses().into_iter()
                .map(|x| x.append(i));

            addresses.extend(children_addresses_iter);
        }

        addresses
    }
}

#[derive(Debug, Default, Clone)]
struct Address {
    inner: Vec<usize>,
}

impl Address {
    fn pop(&mut self) -> Option<usize> {
        self.inner.pop()
    }

    fn append(mut self, i: usize) -> Address {
        self.inner.push(i);
        self
    }

    fn append_to_bottom(mut self, i: usize) -> Address {
        // popped off at the very last
        self.inner.insert(0, i);
        self
    }
}


#[derive(Clone, Copy, Debug)]
enum ModelLabel {
    Add,
    Multiply,
    Negative,
    Divide,
    X,
    Constant(f64),
    Dummy, // dummy value that will panic under everything
}



#[derive(Clone, Copy)]
enum Complexity {
    High,
    Moderate,
    Simple,
}

impl Complexity {
    fn decrement(self) -> Complexity {
        match self {
            Complexity::Simple => unreachable!(),
            Complexity::Moderate => Complexity::Simple,
            Complexity::High => Complexity::Moderate,
        }
    }

    fn decrement_probably(self) -> Complexity {
        let should_decrement = thread_rng().gen_bool(1.0);

        if should_decrement {
            self.decrement()
        } else {
            self // leave intact
        }
    }
}

impl ModelLabel {
    fn random(complexity: Complexity) -> ModelLabel {
        let random_value = thread_rng().gen_range(0, 10);

        match complexity {
            Complexity::Simple => {
                match random_value {
                    0..=5 => ModelLabel::Constant(thread_rng().gen_range(-5, 5) as f64),
                    6..=10 => ModelLabel::X,
                    _ => unreachable!(),
                }
            },
            Complexity::Moderate => {
                match random_value {
                    0..=3 => ModelLabel::Negative,
                    4..=7 => ModelLabel::Multiply,
                    8..=10 => ModelLabel::Add,
                    _ => unreachable!(),
                }

            },

            Complexity::High => {
                ModelLabel::Add
            },
        }
    }

    fn arity(self) -> usize {
        match self {
            ModelLabel::Add => 2,
            ModelLabel::Multiply => 2,
            ModelLabel::Negative => 1,
            ModelLabel::Divide => 2,
            ModelLabel::X => 0,
            ModelLabel::Constant(_) => 0,
            ModelLabel::Dummy => 0,
        }
    }
}

#[derive(Default, Clone)]
struct DataSet {
   points: Vec<DataPoint>,
}

impl DataSet {
    fn add_point(&mut self, x: f64, y: f64) {
        self.points.push(DataPoint { x, y });
    }

    fn from_model(model: &Model) -> DataSet {
        // lets do from -10, 10 with step of 1/2
        let mut dataset = DataSet::default();

        let mut x = -10.0;

        while x < 10.0 {
            match model.evaluate(x) {
                Some(y) => {
                    dataset.add_point(x, y);
                },
                None => {},
            }

            x += 0.5;
        }

        dataset
    }
}

#[derive(Clone, Copy)]
struct DataPoint {
    x: f64,
    y: f64,
}