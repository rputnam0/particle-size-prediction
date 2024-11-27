import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic data for atomization process with SI units
    """
    # Input parameter ranges (converted to SI units)
    pressure = np.random.uniform(100e3, 500e3, n_samples)  # Pa
    flow_rate = np.random.uniform(1/60000, 10/60000, n_samples)  # m³/s
    viscosity = np.random.uniform(0.001, 0.005, n_samples)  # Pa.s
    surface_tension = np.random.uniform(0.02, 0.07, n_samples)  # N/m
    nozzle_diameter = np.random.uniform(0.5e-3, 2.0e-3, n_samples)  # m

    # Physical constants
    density = 1000  # kg/m³ (water)

    # Calculate velocity (m/s)
    velocity = flow_rate / (math.pi * (nozzle_diameter / 2) ** 2)

    # Calculate non-dimensional numbers
    weber = density * velocity ** 2 * nozzle_diameter / surface_tension
    reynolds = density * velocity * nozzle_diameter / viscosity
    ohnesorge = viscosity / np.sqrt(density * surface_tension * nozzle_diameter)

    # Generate particle size based on physics relationships
    mean_particle_size = (
        10e-6 * (weber ** -0.6) * (1 + 2.5 * ohnesorge) * (reynolds ** 0.2) * nozzle_diameter
    )  # m

    # Add realistic noise
    particle_size = mean_particle_size * (1 + 0.1 * np.random.randn(n_samples))

    # Store non-dimensional numbers for physics constraints
    physics_params = {
        'weber': weber,
        'reynolds': reynolds,
        'ohnesorge': ohnesorge,
        'velocity': velocity
    }

    X = np.column_stack([pressure, flow_rate, viscosity, surface_tension, nozzle_diameter])
    y = particle_size.reshape(-1, 1)

    return X, y, physics_params

class MultiScaleAtomizationPINN(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()

        # Bulk flow network
        self.bulk_flow_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Interface network
        self.interface_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Droplet network
        self.droplet_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        bulk_features = self.bulk_flow_network(x)
        interface_features = self.interface_network(x)
        droplet_features = self.droplet_network(x)

        combined_features = torch.cat(
            [bulk_features, interface_features, droplet_features], dim=1
        )

        return self.fusion_layer(combined_features)

def compute_physics_constraints(pred_size, inputs, density=1000.0):
    """
    Compute all physics-based constraints using PyTorch operations
    """
    # Extract inputs (all in SI units)
    pressure = inputs[:, 0]  # Pa
    flow_rate = inputs[:, 1]  # m³/s
    viscosity = inputs[:, 2]  # Pa.s
    surface_tension = inputs[:, 3]  # N/m
    nozzle_diameter = inputs[:, 4]  # m

    # Calculate velocity using PyTorch operations
    velocity = flow_rate / (math.pi * (nozzle_diameter / 2) ** 2)

    # Calculate non-dimensional numbers
    pred_size = pred_size.squeeze()
    weber = density * velocity ** 2 * pred_size / surface_tension
    reynolds = density * velocity * nozzle_diameter / viscosity
    ohnesorge = viscosity / torch.sqrt(density * surface_tension * nozzle_diameter)

    # Weber number constraint (optimal atomization range)
    weber_loss = torch.mean(torch.relu(10 - weber) + torch.relu(weber - 100))

    # Reynolds number constraint (turbulent flow regime preferred)
    reynolds_loss = torch.mean(torch.relu(2000 - reynolds))

    # Ohnesorge number constraint (atomization regime)
    ohnesorge_loss = torch.mean(torch.relu(ohnesorge - 0.1))

    # Mass conservation constraint
    theoretical_volume_flow = flow_rate
    predicted_volume_flow = (
        math.pi * pred_size ** 3 * velocity / (6 * nozzle_diameter ** 2)
    )
    mass_conservation_loss = torch.mean((theoretical_volume_flow - predicted_volume_flow) ** 2)

    return {
        'weber_loss': weber_loss,
        'reynolds_loss': reynolds_loss,
        'ohnesorge_loss': ohnesorge_loss,
        'mass_conservation_loss': mass_conservation_loss
    }

class PhysicsWeightScheduler:
    def __init__(self, initial_weight=0.1, decay_rate=0.95, min_weight=0.01):
        self.weight = initial_weight
        self.decay_rate = decay_rate
        self.min_weight = min_weight

    def step(self, epoch):
        self.weight = max(self.weight * (self.decay_rate ** epoch), self.min_weight)
        return self.weight

def train_model(model, X_train, y_train, X_test, y_test, epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Convert to tensors and move to device
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, verbose=True)
    data_criterion = nn.MSELoss()
    physics_scheduler = PhysicsWeightScheduler()

    train_losses = []
    test_losses = []
    physics_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X_train_tensor)

        # Calculate losses
        data_loss = data_criterion(y_pred, y_train_tensor)
        physics_constraints = compute_physics_constraints(y_pred, X_train_tensor)

        physics_loss = sum(physics_constraints.values())
        physics_weight = physics_scheduler.step(epoch)

        total_loss = data_loss + physics_weight * physics_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test_tensor)
            test_loss = data_criterion(y_test_pred, y_test_tensor)

        train_losses.append(total_loss.item())
        test_losses.append(test_loss.item())
        physics_losses.append(physics_loss.item())

        scheduler.step(test_loss)

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}")
            print(f"Training Loss: {total_loss.item():.6f}")
            print(f"Test Loss: {test_loss.item():.6f}")
            print(f"Physics Loss: {physics_loss.item():.6f}")
            print(f"Physics Weight: {physics_weight:.6f}")
            print("Individual Physics Constraints:")
            for name, value in physics_constraints.items():
                print(f"{name}: {value.item():.6f}")
            print("------------------------")

    return train_losses, test_losses, physics_losses

# Generate data
X, y, physics_params = generate_synthetic_data(1000)

# Scale the data using MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Train model
model = MultiScaleAtomizationPINN()
train_losses, test_losses, physics_losses = train_model(
    model, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
)

# Plotting functions
def plot_training_history(train_losses, test_losses, physics_losses):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(physics_losses, label='Physics Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Physics Constraints Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_predictions(model, X_test_scaled, y_test, scaler_y):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--',
        lw=2
    )
    plt.xlabel('Actual Particle Size (m)')
    plt.ylabel('Predicted Particle Size (m)')
    plt.title('Predicted vs Actual Particle Size')
    plt.grid(True)
    plt.show()

# Plot results
plot_training_history(train_losses, test_losses, physics_losses)
plot_predictions(model, X_test_scaled, y_test, scaler_y)